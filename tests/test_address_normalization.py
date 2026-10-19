"""Real libpostal integration, with independently specified international examples."""
import pytest
from fastapi.testclient import TestClient

from tractor import Address
from tractor.address_normalization import address_parts
from tractor.server import create_app
from tractor.types import ExtractRequest


@pytest.mark.parametrize('text,country,expected', [
    ('Lindenweg 14, 04107 Leipzig', 'DE', {'street': 'Lindenweg 14', 'city': 'Leipzig', 'postcode': '04107'}),
    ('10 Downing Street, London SW1A 2AA', 'GB', {'street': '10 Downing Street', 'city': 'London', 'postcode': 'SW1A 2AA'}),
    ('10 rue de la Paix, 75002 Paris', 'FR', {'street': '10 rue de la Paix', 'city': 'Paris', 'postcode': '75002'}),
    ('350 Fifth Avenue, New York NY 10118', 'US', {'street': '350 Fifth Avenue', 'city': 'New York', 'postcode': '10118'}),
    ('Bedford\nRoad', 'GB', {'street': 'Bedford Road'}),
    ('27 Pear Tree\nStreet', 'GB', {'street': '27 Pear Tree Street'}),
    ('PO Box 406', 'GB', {'street': 'PO Box 406'}),
])
def test_international_components_preserve_spelling(text, country, expected):
    assert address_parts(text, country=country) == expected


def test_missing_components_and_competing_addresses_are_not_completed():
    assert 'postcode' not in address_parts('Lindenweg 14, Leipzig', country='DE')
    assert 'street' not in address_parts('Canterbury', 'street', 'GB')
    assert address_parts('London EC1A 9BU York YO1 7HD', country='GB') == {}


@pytest.mark.parametrize('locality', ['Romford', 'Enfield', 'Twickenham'])
def test_explicit_postal_locality_is_not_replaced_by_administrative_parent(locality):
    assert address_parts(locality, 'city', 'GB') == {'city': locality}
    assert address_parts(locality, 'street', 'GB') == {}


def test_same_public_api_accepts_international_components():
    text = 'Büro: Lindenweg 14, 04107 Leipzig'
    value = 'Lindenweg 14, 04107 Leipzig'
    start = text.index(value)
    class Backend:
        def candidates(self, request):
            if request.text != text:
                return [] if request.anchor else [{}]
            return [{'street': [{'text': value, 'start': start, 'end': len(text)}]}]
    request = ExtractRequest(text=text, fields={key: Address('Büro ' + key, country='DE', component=key, group='office')
                                               for key in ('street', 'city', 'postcode')})
    with TestClient(create_app(extractor=Backend())) as client:
        response = client.post('/v1/extract', json=request.model_dump())
    assert response.status_code == 200
    assert response.json() == {'data': {'street': 'Lindenweg 14', 'city': 'Leipzig', 'postcode': '04107'}, 'issues': []}
