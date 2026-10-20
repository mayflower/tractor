"""International address components from libpostal, preserving selected source text.

No geocoding, generated expansion or inferred country enters the output.
"""
import re
from collections import Counter
from functools import lru_cache
from threading import Lock

from .contact_normalization import clean_text, postal_code
from .field_normalization import Unresolved

_parse_lock = Lock()


def joined_address_spans(text, spans, country=None):
    """Join selected adjacent fragments only when libpostal sees one address.

    No missing words, blank paragraphs, other records or inferred components
    enter the joined evidence. Multiple street/locality components remain separate.
    """
    clusters = []
    for span in sorted(spans, key=lambda s: (s['start'], s['end'])):
        gap = text[clusters[-1][-1]['end']:span['start']] if clusters else None
        if (gap is not None and span['start'] >= clusters[-1][-1]['end']
                and re.fullmatch(r'[ \t,]*(?:\r?\n[ \t,]*)?', gap)):
            clusters[-1].append(span)
        else:
            clusters.append([span])
    result = []
    for cluster in clusters:
        if len(cluster) == 1:
            result.extend(cluster)
            continue
        start, end = cluster[0]['start'], cluster[-1]['end']
        value = text[start:end]
        try:
            counts = Counter(label for _, label in parsed_parts(value, country))
        except Unresolved:
            result.extend(cluster)
            continue
        unique = ('road', 'house_number', 'po_box', 'city', 'postcode', 'country')
        # A partial name or street fragment alone is insufficient to join values.
        if (all(counts[k] <= 1 for k in unique)
                and (counts['road'] or counts['po_box'])
                and (counts['city'] or counts['postcode'])):
            result.append({'start': start, 'end': end, 'text': value})
        else:
            result.extend(cluster)
    return result


@lru_cache(maxsize=256)
def parsed_parts(text, country=None):
    from postal.parser import parse_address

    source = clean_text(text)
    # libpostal returns lowercased segments, not offsets. Recover their original
    # spelling in order; casefold can expand characters, so retain index maps.
    folded, positions = '', []
    for index, char in enumerate(source):
        piece = char.casefold()
        folded += piece
        positions.extend([index] * len(piece))
    with _parse_lock:
        parsed = parse_address(source, country=country.lower() if country else None)
    result, cursor = [], 0
    for value, label in parsed:
        key = clean_text(value).casefold()
        start = folded.find(key, cursor)
        if start < 0:
            raise Unresolved('unresolved_address')
        end = start + len(key)
        if any(c.isalnum() for c in folded[cursor:start]):
            raise Unresolved('unresolved_address')
        result.append((source[positions[start]:positions[end - 1] + 1], label))
        cursor = end
    if not result or any(c.isalnum() for c in folded[cursor:]):
        raise Unresolved('unresolved_address')
    return tuple(result)


def address_parts(text, source_component=None, country='GB'):
    """Parse only supplied evidence. The selected field never dictates the labels."""
    try:
        parts = parsed_parts(text, country)
    except Unresolved:
        return {}
    counts = Counter(label for _, label in parts)
    # The public group describes ONE address. Repeated components are not
    # silently merged or selected; multiple addresses need explicit records.
    if any(counts[label] > 1 for label in ('road', 'house_number', 'po_box', 'city', 'postcode', 'country')):
        return {}
    result = {}
    # A field explicitly requesting the postal locality may contain a suburb
    # or city district (libpostal uses administrative labels). Preserve that
    # stated locality; never replace it with a parent city or promote a district
    # when parsing a full address that already supplies its own city.
    if source_component == 'city' and len(parts) == 1 and parts[0][1] in {'suburb', 'city_district'}:
        result['city'] = parts[0][0]
    street = [value for value, label in parts if label in {'house_number', 'road', 'po_box'}]
    if counts['road'] or counts['po_box']:
        result['street'] = ' '.join(street)
    for value, label in parts:
        if label == 'city':
            result['city'] = value
        elif label == 'postcode':
            try:
                result['postcode'] = postal_code(value, country)
            except Unresolved:
                pass
    return result
