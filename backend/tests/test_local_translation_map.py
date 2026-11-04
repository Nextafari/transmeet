"""NLLB language mapping without loading the model."""

from app.services.local_translation import LocalTranslationEngine


def _map(lang):
    engine = LocalTranslationEngine.__new__(LocalTranslationEngine)
    return engine.map_language(lang)


def test_short_codes_map():
    assert _map("en") == "eng_Latn"
    assert _map("pt") == "por_Latn"
    assert _map("zh-tw") == "zho_Hant"


def test_full_flores_codes_pass_through():
    assert _map("yor_Latn") == "yor_Latn"
    assert _map("swh_Latn") == "swh_Latn"
    assert _map("YOR_latn") == "yor_Latn"


def test_unknown_and_auto_codes_map_to_none():
    assert _map("xx") is None
    assert _map("auto") is None
    assert _map(None) is None
    assert _map("") is None


def _engine(provider):
    engine = LocalTranslationEngine.__new__(LocalTranslationEngine)
    engine.provider = provider
    return engine


def test_is_same_language_nllb_uses_code_map():
    engine = _engine("nllb")
    assert engine.is_same_language("en", "en") is True
    assert engine.is_same_language("en", "EN") is True
    assert engine.is_same_language("ja", "en") is False
    assert engine.is_same_language(None, "en") is False
    assert engine.is_same_language("xx", "xx") is False


def test_is_same_language_marian_compares_plain_codes():
    engine = _engine("marian")
    assert engine.is_same_language("en", "EN") is True
    assert engine.is_same_language("ja", "en") is False
    assert engine.is_same_language(None, "en") is False
    assert engine.is_same_language("xx", "xx") is True
