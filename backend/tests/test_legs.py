"""Translation-leg ordering and fallback."""

import pytest

from app.services.meta.translation import ordered_legs, translate_with_legs


def test_ordered_legs_dedupes_and_validates():
    assert ordered_legs("spark", ["nllb", "spark"]) == ["spark", "nllb"]
    with pytest.raises(ValueError, match="Unknown translation leg"):
        ordered_legs("gemini", [])


def test_empty_legs_rejected():
    with pytest.raises(ValueError, match="No translation leg"):
        ordered_legs(None, [])


class FakeEngine:
    provider = "nllb"

    def __init__(self, same=False, translated="hola-en", fail=None):
        self._same = same
        self._translated = translated
        self._fail = fail
        self.calls = []

    def same_language(self, source, target):
        return self._same

    def translate(self, text, source, target):
        self.calls.append((text, source, target))
        if self._fail is not None:
            raise self._fail
        return self._translated


class FakeSpark:
    def __init__(self, translated="hola-spark", fail=None):
        self._translated = translated
        self._fail = fail
        self.calls = []

    async def translate_text(self, text, target, source_language=None):
        self.calls.append((text, target, source_language))
        if self._fail is not None:
            raise self._fail
        return self._translated


@pytest.mark.asyncio()
async def test_same_language_shortcut_skips_calls():
    engine = FakeEngine(same=True)
    spark = FakeSpark()
    text, leg = await translate_with_legs(
        "hello",
        source_language="en",
        target_language="en",
        primary="nllb",
        fallbacks=["spark"],
        local_engine=engine,
        spark_client=spark,
    )
    assert (text, leg) == ("hello", "nllb")
    assert engine.calls == [] and spark.calls == []


@pytest.mark.asyncio()
async def test_primary_leg_serves():
    engine = FakeEngine()
    text, leg = await translate_with_legs(
        "hola",
        source_language="es",
        target_language="en",
        primary="nllb",
        fallbacks=["spark"],
        local_engine=engine,
    )
    assert (text, leg) == ("hola-en", "nllb")
    assert engine.calls == [("hola", "es", "en")]


@pytest.mark.asyncio()
async def test_fallback_leg_used_when_primary_fails():
    engine = FakeEngine(fail=RuntimeError("nllb down"))
    spark = FakeSpark()
    text, leg = await translate_with_legs(
        "hola",
        source_language="es",
        target_language="en",
        primary="nllb",
        fallbacks=["spark"],
        local_engine=engine,
        spark_client=spark,
    )
    assert (text, leg) == ("hola-spark", "spark")


@pytest.mark.asyncio()
async def test_all_legs_failing_raises():
    engine = FakeEngine(fail=RuntimeError("nllb down"))
    with pytest.raises(RuntimeError):
        await translate_with_legs(
            "hola",
            source_language="es",
            target_language="en",
            primary="nllb",
            fallbacks=[],
            local_engine=engine,
            spark_client=None,
        )
