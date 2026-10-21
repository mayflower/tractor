# Extract and normalize typed fields

Use `POST /v1/extract` or `Client.extract` for every extraction. Each field has
one required type, which determines its processing. There is no separate
processing map or document endpoint. Unknown types and incompatible options
are rejected before inference.

```python
from tractor import Client, Date, Identifier, Money

with Client("http://localhost:8000") as client:
    result = client.extract(
        "Rechnung R-0042 vom 24.10.2026 über 1.234,50 EUR.",
        fields={
            "reference": Identifier("Rechnungsnummer"),
            "date": Date("Rechnungsdatum", date_order="DMY"),
            "amount": Money("Rechnungsbetrag", currency="EUR"),
        },
        locale="de-DE",
    )
    print(result.data)
```

When the corresponding evidence is extracted, the final values are
`reference: "R-0042"`, `date: "2026-10-24"`, and `amount: 1234.50`.
Descriptions identify the requested role; types select normalization.
A description saying "date" does not change a field declared as `text`.

| Type | Parameters | Processing |
|---|---|---|
| `text` | description | Source text; no inferred semantic rewriting |
| `quantity` | required `unit`, optional `locale` | Parse and convert compatible units to the requested unit |
| `number` | optional `locale` | Decimal number; ambiguous separators and ranges produce issues |
| `money` | required `currency`, optional `locale` | Decimal amount; validate source currency, no exchange-rate conversion |
| `date` | `date_order`, `allow_partial` | ISO date; numeric month/year accepted with partial precision enabled |
| `time` | none | ISO clock time, preserving minute/second precision and explicit offset |
| `datetime` | `date_order` | Full date and time; no assumed timezone |
| `duration` | none | ISO duration; calendar months remain months |
| `interval` | `date_order` | ISO start/end; both boundaries must be explicit |
| `percent` | optional `locale` | Decimal percentage points: `4.2 %` becomes `4.2` |
| `iban` | none | Compact uppercase IBAN with local format/checksum validation |
| `url` | none | Normalize scheme/host and IDNA; preserve path, query and fragment |
| `person` | none | NFC Unicode and whitespace normalization; preserve name parts and titles |
| `organization` | none | NFC Unicode and whitespace normalization; preserve legal form |
| `place` | none | NFC Unicode and whitespace normalization; no place lookup |
| `phone` | `country` for national numbers | Validate and format internationally; preserve extensions |
| `email` | none | Validate syntax and normalize domain; no DNS lookup |
| `postcode` | required `country` | Validate country-specific format; retain leading zeros |
| `address` | optional `country`; components require `country`, `component`, `group` | Full address formatting; international component parsing with libpostal |
| `identifier` | `remove_spaces`, `uppercase`; optional `format`: `isbn`, `isbn13`, `ean`, `vat` (`vat` requires `country`) | Generic formatting or local standard-format/checksum validation; `isbn13` converts valid ISBN-10 and ISBN-13 to ISBN-13 |
| `list` | required `separator`: `newline` or `semicolon` | Split the extracted text at the declared separator |

Quantity and number processing use Pint, Babel and Decimal. Date processing
uses Python datetime and dateparser, supporting ISO dates and English/German month
names. Numeric dates require an explicit order unless ISO. Partial dates require
`allow_partial: true`. For example, `03/2019` becomes `2019-03`, without
inventing a day. A `date` field requests only the date even if its detected span
also includes a valid time; use `datetime` when the time is required.
Relative dates and unspecified timezones are never resolved from the current clock.
Money requires source currency evidence. Dollar symbols require an explicit
supported dollar currency. No missing facts are supplied.
Currency names and symbols come from Babel's CLDR data for the supported currencies
and DE/EN locales. For example, `US$ 1,250.00` with `currency="USD", locale="en-US"`
and `1.250,00 US-Dollar` with `currency="USD", locale="de-DE"` both produce `1250.00`.
`CA$` conflicts with `currency="USD"`; a locale never supplies a missing currency.

## Standard identifiers

```python
from tractor import Client, Identifier

with Client() as client:
    result = client.extract(
        "ISBN 978-0-471-11709-4; EAN 4006381333931; USt-ID DE136695976.",
        {
            "book": Identifier("ISBN", format="isbn"),
            "barcode": Identifier("EAN", format="ean"),
            "vat": Identifier("USt-ID", format="vat", country="DE"),
        },
    )
```

When the corresponding evidence is selected, values are `9780471117094`,
`4006381333931` and `136695976`. These formats use
[python-stdnum](https://arthurdejong.org/python-stdnum/) for local validation and
compact representation. ISBN-10 stays ISBN-10; no conversion to ISBN-13 occurs.
VAT output is the country's library-normalized identifier (for Germany, nine
digits without `DE`); its country remains explicit in the field specification.
VAT country support follows the installed library; unsupported countries yield
`unsupported_identifier_country`. Bad checksums or formats yield `invalid_identifier`.
Validation does not query registries or establish that a business exists.

The format determines normalization; combining it with `remove_spaces` or
`uppercase` is rejected. Without `format`, identifiers retain the existing generic
behavior. A field name such as `isbn` never implicitly activates checksum validation.

Use `Identifier("Book ISBN", format="isbn13")` when both ISBN-10 and ISBN-13
mentions should resolve to the same ISBN-13 value. For example, `0-306-40615-2`
and `978-0-306-40615-7` both become `9780306406157`. The existing `isbn` format
validates and compacts the supplied format without converting its length.

## Address components

```python
from tractor import Address, Client

with Client() as client:
    result = client.extract(
        "Registered office: 12 Cock Lane, London EC1A 9BU.",
        {
            part: Address(
                f"Registered office {part}",
                country="GB", component=part, group="office",
            )
            for part in ("street", "city", "postcode")
        },
    )
```

Only detected evidence in the same address group and record can supply another
component. For example, a detected `London EC1A 9BU` span can supply both town
and postcode. Components use [libpostal](https://github.com/openvenues/libpostal)
through its [Python bindings](https://github.com/openvenues/pypostal), with
country-specific postal validation from [google-i18n-address](https://github.com/mirumee/google-i18n-address).
The parser is statistical and can be wrong; original spelling is retained and
repeated competing components are not silently merged. Countries are explicit.
No expansion candidate is arbitrarily chosen and no missing component is supplied.
No geocoding or postal directory is used. Without `component`,
`address` normalizes the full extracted address: Unicode, whitespace, line breaks
and comma spacing. This works independently of country but does not parse or
validate deliverability. For example, use `country="DE"` for components of
`Lindenweg 14, 04107 Leipzig`, or `country="FR"` for a French address.

## Scalars, lists and repeated records

Fields default to `cardinality: "optional_one"`: a value or null. Distinct
competing normalized values are all returned as an array, in source order,
together with an `ambiguous_value` issue. Nothing is dropped and no candidate is picked.
`required_one` additionally reports a missing required value.
`unique_list` returns all distinct normalized values, or an empty list.
`one_or_more` additionally reports an empty result. The `list` type always
uses list cardinality.

Repeated mentions that normalize to the same value do not create scalar
ambiguity. The public response contains only `data` and `issues`.

For repeated records, use the same request with `many: true` and `anchor`
naming a text, name, place or identifier field that identifies each explicitly described record.
Records remain separate even when some of their values are equal. This does
not reconstruct tables or infer relationships.

```json
{
  "text": "Atlas weighs 12 kg. Boreal weighs 8 kg.",
  "fields": {
    "name": {"type": "text", "description": "Product name"},
    "weight": {"type": "quantity", "description": "Product weight", "unit": "kg"}
  },
  "anchor": "name",
  "many": true
}
```

## Evaluation

All normalization runs before the API returns values. Evaluation constructs
the same typed requests and calls the same extraction service. Its corpus adapter
only reshapes scalar/list outputs for comparison; it cannot repair predicted
values using annotations.
For numeric field types, final annotations compare numerically: `1250.00` and
`1250` are equal. Source-format parsing is not applied to expected annotations;
they must already express final values. Text and identifiers retain their lexical
distinctions, including leading zeros.

Legacy corpus fields without an explicit processing contract are represented as
`text`, not silently reclassified using their names or gold answers. This is
visible migration debt: it does not establish normalized coverage for dates,
addresses or other semantic types. Quality must be re-evaluated after a corpus
contract audit; previous results from the separate document path are not scores
for this unified API.

Callers must inspect `issues` for failed normalization, missing values and
ambiguity. Source offsets and model diagnostics are internal; the API does not
expose a source-evidence option.

## Contact fields

```python
from tractor import Address, Client, Email, Person, Phone, Place, Postcode

with Client() as client:
    result = client.extract(
        "Kontakt: José van Dijk, Telefon 030 901820, Person@EXAMPLE.ORG. "
        "Anschrift: Lindenweg 14, 04107 Leipzig.",
        {
            "name": Person("Kontaktperson"),
            "phone": Phone("Telefonnummer", country="DE"),
            "email": Email("E-Mail-Adresse"),
            "postcode": Postcode("Postleitzahl", country="DE"),
            "city": Place("Ort der Anschrift"),
            "address": Address("Vollständige Anschrift"),
        },
    )
    print(result.data, result.issues)
```

Phone processing uses [phonenumbers](https://github.com/daviddrysdale/python-phonenumbers).
National numbers require a country; international numbers with `+` do not.
The output is E.164, with `;ext=42` appended when an extension is present.
Invalid numbers remain unresolved; formatting validation does not establish reachability.

Email processing uses [email-validator](https://github.com/JoshData/python-email-validator)
with deliverability checks disabled. Domain case and Unicode are normalized;
local-part case and plus tags are preserved. No address is contacted.

Postal processing uses the bundled country rules from
[google-i18n-address](https://github.com/mirumee/google-i18n-address).
For example, `ec1a9bu` becomes `EC1A 9BU`, `k1a0b1` becomes `K1A 0B1`,
and German `01067` keeps its leading zero. Unknown countries or countries
without postal-code validation rules produce an issue. This checks format,
not whether a postcode exists or matches a town. A larger detected field span,
such as `04107 Leipzig`, can supply its one explicit postcode. Multiple distinct
postcodes in the span remain ambiguous; no surrounding document search occurs.

Names retain spelling, accents, titles, qualifiers and legal forms. Their
deduplication/evaluation keys ignore case, preserving the first normalized
spelling for output. `LEIPZIG` and `Leipzig` compare equally; `Frankfurt` and
`Frankfurt am Main`, `Müller` and `Muller`, or `ACME GmbH` and `ACME AG`
remain distinct. Full addresses use the same case-insensitive comparison after
format normalization. Email local parts and generic identifiers stay case-sensitive.


## Dates, times and intervals

```python
from tractor import Client, Datetime, Duration, Interval, Time

with Client() as client:
    result = client.extract(
        "Prüfung: 14.10.2026, 07:45 Uhr. Zeitfenster: 09:00–11:00. "
        "Laufzeit: 3 Monate.",
        {
            "inspection": Datetime("Zeitpunkt der Prüfung", date_order="DMY"),
            "window": Interval("Zeitfenster"),
            "duration": Duration("Laufzeit"),
        },
    )
    print(result.data)
```

When the corresponding evidence is selected, the output values are
`2026-10-14T07:45`, `09:00/11:00` and `P3M`.
Minute precision stays minute precision. Explicit numeric timezone offsets, Z,
UTC and GMT are preserved as offsets; ambiguous timezone abbreviations such as
CST are unresolved. Local times remain local and have no invented offset.
Named dates such as `October 14th, 2026` and `14th of October 2026` are parsed by
[dateparser](https://dateparser.readthedocs.io/en/latest/) in absolute, strict mode.
Additional source-component checks reject two-digit years, missing components,
approximations and silently reinterpreted dates. `about 14 October 2026` remains
unresolved. Numeric date order is enforced directly, without library fallback.

Intervals accept two full dates, two times or two datetimes. Both endpoints
must have the same type and compatible timezone information, and the end must
not precede the start. `23:00–01:00` does not invent a next day.
`fiscal year ended 31 March 2025` does not invent a start date.

Duration accepts ISO durations and numeric English/German unit phrases such as
`90 days`, `3 calendar months` and `2 Stunden und 30 Minuten`. It does not
convert months to days or calculate deadlines. Spelled-out numbers, approximations,
ranges, mixed weeks/other units and repeated units remain unresolved.
ISO parsing uses [isoduration](https://pypi.org/project/isoduration/) with Decimal
components: `P1,5Y` becomes `P1.5Y`, and `PT0.000000000123S` keeps its full precision.
Explicit zero components are retained. Fractions are allowed only on the smallest
stated unit; exponents and negative durations remain unsupported.

## Numbers, money and percentages

`number` accepts explicit count labels and common count nouns:
`n = 186`, `128 participants`, `142 Mitarbeitende`.
Counts must be nonnegative integers. Bare numbers retain ordinary signed/decimal
semantics. Multiple numbers and arithmetic such as `12 × 5` are not evaluated.
An explicit `n =` label can carry a count description and a parenthesized
breakdown: `n = 60 specimens (12 formulations × 5 replicates)` returns `60`.
The stated count is read directly; the subgroup numbers are not multiplied.
Without that label, `48 (20 controls)` remains unresolved. Alternative counts,
approximations and multiple `n =` labels also remain unresolved.

`money` accepts currency codes, supported currency names, symbols and common
magnitude words. `240 Millionen Euro` becomes `240000000` EUR.
The word `billion` requires an explicit locale because its English and German
meanings differ. Known price labels such as MSRP are removed.
The type requests the explicitly stated amount: `£3,850 per month` becomes
`3850` GBP, never an annualized figure. Extract the period into another requested
field if needed. `per annum`, `per visit` and `per inspection` are also supported
amount qualifiers. `£58,000 per annum` needs an explicit numeric locale to resolve
the comma; its currency does not supply one. Tax qualifiers such as `plus VAT`, ranges, approximations,
multiple currencies and arithmetic remain unresolved.

`percent` returns percentage points, not a fraction: `4,2 %` with de-DE
becomes `4.2`. A bare `0.042` is not silently interpreted as a percentage.
Basis points, ranges and inferred ratios are unsupported.

## IBANs and URLs

`iban` removes formatting spaces, preserves characters/leading zeroes and checks
the country-specific structure and checksum using
[python-stdnum](https://arthurdejong.org/python-stdnum/).
No bank is contacted. An invalid printed checksum is a validation issue;
the parser does not repair it or imply that an account exists.

`url` accepts HTTP(S) addresses and schemeless hostnames with optional paths.
For example, `HTTPS://EXAMPLE.ORG/SomePath?q=Hi` becomes
`https://example.org/SomePath?q=Hi`.
A schemeless `www.example.org` remains schemeless. IDNA hostnames are normalized;
path/query case, query ordering and fragments are retained. No URL is fetched.
Credentials, unsupported schemes, malformed hosts/ports, invalid percent escapes,
whitespace and control characters are rejected.

```json
{
  "text": "IBAN: GB29 NWBK 6016 1331 9268 19. Rabatt: 4,2 %. Website: https://EXAMPLE.ORG/Shop.",
  "fields": {
    "account": {"type": "iban", "description": "IBAN"},
    "discount": {"type": "percent", "description": "Rabatt"},
    "website": {"type": "url", "description": "Website"}
  },
  "locale": "de-DE"
}
```
