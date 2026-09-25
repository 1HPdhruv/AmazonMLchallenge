from src.preprocessing.address import norm_country, parse_full_address, parse_street
from src.preprocessing.normalize import basic, fold, legal_form, name_key, name_tokens


def test_basic_and_fold_are_additive():
    assert basic("  Bäckerei  MÜLLER & Söhne, GmbH. ") == "bäckerei müller and söhne gmbh"
    assert fold(basic("Straße Café")) == "strasse cafe"


def test_name_key_is_order_insensitive_and_strips_legal_forms():
    assert name_key("Eagle Harbor Imports LLC") == name_key("HARBOR EAGLE IMPORTS")
    assert "llc" not in name_tokens("Acme Intl LLC")
    assert "international" in name_tokens("Acme Intl LLC")
    assert legal_form("Acme Pvt Ltd") == "pvt ltd"


def test_name_tokens_never_empty_for_legal_only_names():
    assert name_tokens("The Co") != []


def test_parse_street_us_unit_and_house():
    d = parse_street("2243 Fleurs Street, Suite 120")
    assert d["house"] == "2243" and d["unit"] == "120" and d["street_core"] == "fleurs"
    assert parse_street("12 Main St #40")["unit"] == "40"


def test_parse_street_german_compound():
    d = parse_street("Mozartstraße 12")
    assert d["house"] == "12" and d["street_core"] == "mozart"


def test_parse_full_address_variants():
    us = parse_full_address("21 Hill Ave, Houston, TX 46485, USA")
    assert us["country"] == "US" and us["postal"] == "46485" and us["city"] == "Houston" and us["region"] == "TX"
    br = parse_full_address("Rua Victoria, 6, 92906-675, Curitiba, Brasil")
    assert br["address"] == "Rua Victoria, 6" and br["postal"] == "92906-675" and br["country"] == "BR"
    assert parse_full_address(None) == {}
    assert norm_country("Deutschland") == "DE"
