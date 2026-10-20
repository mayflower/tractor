"""Explicit public field contracts, independent of models and annotated answers."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Country = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]


class Processing(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    kind: Literal['literal', 'quantity', 'address', 'number', 'money', 'date', 'identifier', 'list', 'person', 'organization', 'place', 'phone', 'email', 'postcode', 'time', 'datetime', 'interval', 'duration', 'percent', 'iban', 'url']
    unit: str | None = None
    country: Country | None = None
    component: Literal['full', 'street', 'city', 'postcode'] | None = None
    group: str | None = Field(default=None, min_length=1, max_length=64)
    locale: Literal['de-DE', 'en-US', 'en-GB'] | None = None
    currency: Literal['EUR', 'GBP', 'USD', 'CAD', 'AUD', 'CHF'] | None = None
    date_order: Literal['DMY', 'MDY', 'YMD'] | None = None
    allow_partial: bool = False
    include_adjacent_year: bool = False
    separator: Literal['newline', 'semicolon'] | None = None
    prefixes: list[str] = Field(default_factory=list, max_length=16)
    remove_spaces: bool = False
    uppercase: bool = False
    format: Literal['isbn', 'isbn13', 'ean', 'vat'] | None = None
    cardinality: Literal['unique_list', 'optional_one', 'required_one', 'one_or_more'] = 'unique_list'

    @model_validator(mode='after')
    def valid(self):
        from .types import UNITS
        allowed = {
            'literal': {'include_adjacent_year'}, 'quantity': {'unit', 'locale'},
            'address': {'country', 'component', 'group'}, 'number': {'locale'},
            'money': {'locale', 'currency'}, 'date': {'date_order', 'allow_partial'},
            'time': set(), 'datetime': {'date_order'}, 'interval': {'date_order'},
            'duration': set(), 'percent': {'locale'}, 'iban': set(), 'url': set(),
            'person': set(), 'organization': set(), 'place': set(),
            'phone': {'country'}, 'email': set(), 'postcode': {'country'},
            'identifier': {'remove_spaces', 'uppercase', 'format', 'country'}, 'list': {'separator'},
        }
        if any(not p or len(p) > 64 for p in self.prefixes) or (self.prefixes and self.kind in {'quantity', 'address'}):
            raise ValueError('Literal label prefixes require nonempty bounded labels and supported processing')
        options = ('unit', 'country', 'component', 'group', 'locale', 'currency',
                   'date_order', 'allow_partial', 'include_adjacent_year', 'separator', 'remove_spaces', 'uppercase', 'format')
        if any(getattr(self, k) not in (None, False) and k not in allowed[self.kind] for k in options):
            raise ValueError('Options do not match the declared processing kind')
        if self.kind == 'quantity' and self.unit not in UNITS:
            raise ValueError('Quantity requires a supported target unit')
        if self.kind == 'address':
            if self.component not in (None, 'full') and (not self.country or not self.group):
                raise ValueError('Address components require explicit country and group')
            if self.component in (None, 'full') and self.group:
                raise ValueError('Full addresses do not share evidence groups')
        if self.kind == 'postcode' and not self.country:
            raise ValueError('Postal code requires explicit country')
        if self.kind == 'money' and not self.currency:
            raise ValueError('Money requires an explicit output currency; no conversion is performed')
        if self.kind == 'list' and self.separator is None:
            raise ValueError('List requires an explicit separator')
        if self.kind == 'identifier':
            if (self.format == 'vat') != bool(self.country):
                raise ValueError('Only VAT identifiers require an explicit country')
            if self.format and (self.remove_spaces or self.uppercase):
                raise ValueError('Standard identifier formats determine their own normalization')
        return self
