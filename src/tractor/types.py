"""Public field specifications and strict request validation; no ML imports."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

UNITS = frozenset(
    "mm cm m km in ft yd mile nmi mg g kg lb ms s min h mL L m3 mm2 cm2 m2 m/s km/h mph nm um ug Pa kPa MPa K degC degF".split()
)
Key = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]
Description = Annotated[str, StringConstraints(min_length=1, max_length=240)]


class FieldBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    description: Description
    cardinality: Literal["optional_one", "required_one", "unique_list", "one_or_more"] = "optional_one"
    prefixes: list[str] = Field(default_factory=list, max_length=16)

    def __init__(self, description: str, **kwargs):
        super().__init__(description=description, **kwargs)

    def processing_rule(self):
        from .field_contracts import Processing
        options = self.model_dump(exclude={"type", "description"})
        return Processing(kind="literal" if self.type == "text" else self.type, **options)

    @model_validator(mode="after")
    def valid_processing(self):
        self.processing_rule()
        return self


class Text(FieldBase):
    type: Literal["text"] = "text"
    include_adjacent_year: bool = False


class Quantity(FieldBase):
    type: Literal["quantity"] = "quantity"
    unit: str
    locale: Literal["de-DE", "en-US", "en-GB"] | None = None


class Date(FieldBase):
    type: Literal["date"] = "date"
    date_order: Literal["DMY", "MDY", "YMD"] | None = None
    allow_partial: bool = False


class Time(FieldBase):
    type: Literal["time"] = "time"


class Datetime(FieldBase):
    type: Literal["datetime"] = "datetime"
    date_order: Literal["DMY", "MDY", "YMD"] | None = None


class Interval(FieldBase):
    type: Literal["interval"] = "interval"
    date_order: Literal["DMY", "MDY", "YMD"] | None = None


class Duration(FieldBase):
    type: Literal["duration"] = "duration"


class Percent(FieldBase):
    type: Literal["percent"] = "percent"
    locale: Literal["de-DE", "en-US", "en-GB"] | None = None


class Iban(FieldBase):
    type: Literal["iban"] = "iban"


class Url(FieldBase):
    type: Literal["url"] = "url"


class Number(FieldBase):
    type: Literal["number"] = "number"
    locale: Literal["de-DE", "en-US", "en-GB"] | None = None


class Money(FieldBase):
    type: Literal["money"] = "money"
    currency: Literal["EUR", "GBP", "USD", "CAD", "AUD", "CHF"]
    locale: Literal["de-DE", "en-US", "en-GB"] | None = None


class Address(FieldBase):
    type: Literal["address"] = "address"
    country: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")] | None = None
    component: Literal["full", "street", "city", "postcode"] = "full"
    group: str | None = Field(default=None, min_length=1, max_length=64)


class Person(FieldBase):
    type: Literal["person"] = "person"


class Organization(FieldBase):
    type: Literal["organization"] = "organization"


class Place(FieldBase):
    type: Literal["place"] = "place"


class Phone(FieldBase):
    type: Literal["phone"] = "phone"
    country: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")] | None = None


class Email(FieldBase):
    type: Literal["email"] = "email"


class Postcode(FieldBase):
    type: Literal["postcode"] = "postcode"
    country: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]



class Identifier(FieldBase):
    type: Literal["identifier"] = "identifier"
    format: Literal["isbn", "ean", "vat"] | None = None
    country: Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")] | None = None
    remove_spaces: bool = False
    uppercase: bool = False


class List(FieldBase):
    type: Literal["list"] = "list"
    separator: Literal["newline", "semicolon"]
    cardinality: Literal["unique_list", "one_or_more"] = "unique_list"


FieldSpec = Annotated[Text | Quantity | Date | Number | Money | Address | Identifier | List | Person | Organization | Place | Phone | Email | Postcode | Time | Datetime | Interval | Duration | Percent | Iban | Url,
                      Field(discriminator="type")]


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: Annotated[str, StringConstraints(min_length=1)]
    fields: dict[Key, FieldSpec] = Field(min_length=1, max_length=32)
    locale: Literal["de-DE", "en-US", "en-GB"] | None = None
    many: bool = False
    anchor: str | None = None

    @model_validator(mode="after")
    def validate_anchor(self):
        if not self.text.strip():
            raise ValueError("Text must not be blank")
        if self.many and self.anchor is None:
            raise ValueError("many requires anchor")
        if self.anchor is not None and not isinstance(self.fields.get(self.anchor), (Text, Person, Organization, Place, Identifier)):
            raise ValueError("anchor must name a text, name, place or identifier field")
        if self.anchor and (self.fields[self.anchor].prefixes or getattr(self.fields[self.anchor], "include_adjacent_year", False)):
            raise ValueError("Record anchor must preserve literal identity")
        return self


class TractorError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
