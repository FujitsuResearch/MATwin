import pytest

from isuelogit.geographer import california_county_fips


@pytest.mark.parametrize('name, fips', [
    ('Fresno', '06019'),
    ('Los Angeles', '06037'),
    ('San Diego', '06073'),
    ('Santa Clara', '06085'),
    ('San Francisco', '06075'),
])
def test_known_counties(name, fips):
    assert california_county_fips(name) == fips


@pytest.mark.parametrize('name', ['fresno', 'FRESNO', ' Fresno ', 'Fresno County', 'fresno  county'])
def test_common_spellings(name):
    assert california_county_fips(name) == '06019'


@pytest.mark.parametrize('name', ['Atlantis', '', None, 19])
def test_unknown_returns_none(name):
    assert california_county_fips(name) is None
