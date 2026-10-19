"""Local format validation; never resolve URLs or contact banking services."""
import ipaddress
import re
from urllib.parse import urlsplit, urlunsplit

from .field_normalization import Unresolved


def standard_identifier(value, format, country=None):
    """Validate explicit standard formats locally, including their checksums."""
    from stdnum import ean, get_cc_module, isbn
    from stdnum.exceptions import ValidationError

    parser = {'isbn': isbn, 'ean': ean}.get(format)
    if format == 'vat':
        parser = get_cc_module(country.lower(), 'vat')
    if parser is None:
        raise Unresolved('unsupported_identifier_country')
    # Library compactors intentionally tolerate punctuation. Do not let them
    # erase separators between competing values or unrelated surrounding text.
    if not re.fullmatch(r'[A-Za-z0-9\s.-]+', value, flags=re.ASCII):
        raise Unresolved('invalid_identifier')
    try:
        return parser.validate(value)
    except ValidationError as exc:
        raise Unresolved('invalid_identifier') from exc


def iban(value):
    from stdnum import iban as parser
    from stdnum.exceptions import ValidationError
    value = re.sub(r"^IBAN\s*:\s*", "", value, flags=re.I)
    if not re.fullmatch(r"[A-Za-z]{2}[0-9]{2}[A-Za-z0-9\s]+", value):
        raise Unresolved("invalid_iban")
    try:
        return parser.validate(value)
    except ValidationError as exc:
        raise Unresolved("invalid_iban") from exc


def url(value):
    import idna
    if re.search(r"\s|[\x00-\x1f\x7f\\\\]", value) or re.search(r"%(?![0-9A-Fa-f]{2})", value):
        raise Unresolved("invalid_url")
    has_scheme = bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value))
    protocol_relative = value.startswith("//")
    try:
        parts = urlsplit(value if has_scheme or protocol_relative else "//" + value)
        if (has_scheme and parts.scheme.lower() not in {"http", "https"}) or not parts.hostname:
            raise ValueError("unsupported URL")
        if parts.username is not None or parts.password is not None:
            raise ValueError("userinfo is unsupported")
        host = parts.hostname
        try:
            address = ipaddress.ip_address(host)
            host = "[" + address.compressed + "]" if address.version == 6 else address.compressed
        except ValueError:
            if "." not in host or re.fullmatch(r"[0-9.]+", host):
                raise ValueError("invalid hostname") from None
            host = idna.encode(host.lower(), uts46=True).decode("ascii")
        port = parts.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("invalid port")
        netloc = host + (":" + str(port) if port is not None else "")
        result = urlunsplit((parts.scheme.lower(), netloc, parts.path, parts.query, parts.fragment))
        # urlunsplit removes empty query/fragment delimiters; retain them.
        if "?" in value.split("#", 1)[0] and not parts.query:
            pos = result.find("#")
            result = result + "?" if pos < 0 else result[:pos] + "?" + result[pos:]
        if value.endswith("#") and not parts.fragment:
            result += "#"
        if not has_scheme and not protocol_relative:
            result = result.removeprefix("//")
        return result
    except (ValueError, idna.IDNAError) as exc:
        raise Unresolved("invalid_url") from exc
