"""Public field specifications and strict request validation; no ML imports."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

UNITS = frozenset(
    "mm cm m km in ft yd mile nmi mg g kg lb ms s min h mL L m3 mm2 cm2 m2 m/s km/h mph".split()
)
Key = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]
Description = Annotated[str, StringConstraints(min_length=1, max_length=240)]


class Text(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    type: Literal["text"] = "text"
    description: Description

    def __init__(self, description: str, **kwargs):
        super().__init__(description=description, **kwargs)


class Quantity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    type: Literal["quantity"] = "quantity"
    description: Description
    unit: str

    def __init__(self, description: str, *, unit: str, **kwargs):
        super().__init__(description=description, unit=unit, **kwargs)

    @model_validator(mode="after")
    def valid_unit(self):
        if self.unit not in UNITS:
            raise ValueError("Unsupported target unit")
        return self


FieldSpec = Annotated[Text | Quantity, Field(discriminator="type")]


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: Annotated[str, StringConstraints(min_length=1)]
    fields: dict[Key, FieldSpec] = Field(min_length=1, max_length=32)
    locale: Literal["de-DE", "en-US", "en-GB"] | None = None
    many: bool = False
    anchor: str | None = None
    include_details: bool = False

    @model_validator(mode="after")
    def validate_anchor(self):
        if not self.text.strip():
            raise ValueError("Text must not be blank")
        if self.many and self.anchor is None:
            raise ValueError("many requires anchor")
        if self.anchor is not None and not isinstance(self.fields.get(self.anchor), Text):
            raise ValueError("anchor must name a Text field")
        return self


class TractorError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
