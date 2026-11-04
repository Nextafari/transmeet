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

    def __init__(self, same=False, translated="hola-en", fail=None, provider="nllb"):
        self.provider = provider
        self._same = same
        self._translated = translated
        self._fail = fail
        self.calls = []

    def same_language(self, source, target):
        return self._same

    def is_same_language(self, source, target):
        if self.provider == "marian":
            src = str(source or "").strip().lower()
            tgt = str(target or "").strip().lower()
            return bool(src and src == tgt)
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
        self.timeouts = []

    async def translate_text(
        self, text, target, source_language=None, timeout=None
    ):
        self.calls.append((text, target, source_language))
        self.timeouts.append(timeout)
        if self._fail is not None:
            raise self._fail
        return self._translated


class FakeBaseten:
    def __init__(self, translated="hola-baseten", fail=None):
        self._translated = translated
        self._fail = fail
        self.calls = []
        self.timeouts = []

    async def translate_text(
        self, text, target, source_language=None, timeout=None
    ):
        self.calls.append((text, target, source_language))
        self.timeouts.append(timeout)
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


def test_ordered_legs_accepts_baseten():
    assert ordered_legs("baseten", ["nllb"]) == ["baseten", "nllb"]


@pytest.mark.asyncio()
async def test_baseten_leg_serves():
    baseten = FakeBaseten()
    text, leg = await translate_with_legs(
        "hola",
        source_language="es",
        target_language="en",
        primary="baseten",
        fallbacks=["nllb"],
        baseten_client=baseten,
    )
    assert (text, leg) == ("hola-baseten", "baseten")
    assert baseten.calls == [("hola", "en", "es")]


@pytest.mark.asyncio()
async def test_baseten_leg_falls_back_when_unconfigured():
    engine = FakeEngine()
    text, leg = await translate_with_legs(
        "hola",
        source_language="es",
        target_language="en",
        primary="baseten",
        fallbacks=["nllb"],
        local_engine=engine,
        baseten_client=None,
    )
    assert (text, leg) == ("hola-en", "nllb")


@pytest.mark.asyncio()
async def test_repeated_phrase_run_collapses():
    engine = FakeEngine(
        translated=(
            "So I went to Chile, and I went to Chile, and I went to Chile, "
            "and I went to Chile, and I went to Chile"
        )
    )
    text, leg = await translate_with_legs(
        "チリに行きました",
        source_language="ja",
        target_language="en",
        primary="nllb",
        fallbacks=[],
        local_engine=engine,
    )
    assert leg == "nllb"
    assert text == "So I went to Chile, and I went to Chile"


@pytest.mark.asyncio()
async def test_repeated_sentence_run_collapses():
    engine = FakeEngine(
        translated="I went to Chile. " * 6,
    )
    text, _leg = await translate_with_legs(
        "チリに行きました",
        source_language="ja",
        target_language="en",
        primary="nllb",
        fallbacks=[],
        local_engine=engine,
    )
    assert text == "I went to Chile."


@pytest.mark.asyncio()
async def test_short_repetition_left_alone():
    engine = FakeEngine(translated="No, no, no!")
    text, _leg = await translate_with_legs(
        "いいえ",
        source_language="ja",
        target_language="en",
        primary="nllb",
        fallbacks=[],
        local_engine=engine,
    )
    assert text == "No, no, no!"


@pytest.mark.asyncio()
async def test_marian_same_language_skips_model():
    engine = FakeEngine(provider="marian", translated="GARBAGE")
    spark = FakeSpark()
    text, _leg = await translate_with_legs(
        "hello",
        source_language="en",
        target_language="EN",
        primary="nllb",
        fallbacks=["spark"],
        local_engine=engine,
        spark_client=spark,
    )
    assert text == "hello"
    assert engine.calls == [] and spark.calls == []


@pytest.mark.asyncio()
async def test_marian_different_languages_translate():
    engine = FakeEngine(provider="marian", translated="konnichiwa")
    text, leg = await translate_with_legs(
        "hello",
        source_language="en",
        target_language="ja",
        primary="nllb",
        fallbacks=[],
        local_engine=engine,
    )
    assert (text, leg) == ("konnichiwa", "nllb")
    assert engine.calls == [("hello", "en", "ja")]


@pytest.mark.asyncio()
async def test_timeout_forwarded_to_api_legs():
    spark = FakeSpark()
    text, leg = await translate_with_legs(
        "hola",
        source_language="es",
        target_language="en",
        primary="spark",
        fallbacks=[],
        spark_client=spark,
        timeout=7.0,
    )
    assert (text, leg) == ("hola-spark", "spark")
    assert spark.timeouts == [7.0]

    baseten = FakeBaseten()
    text, leg = await translate_with_legs(
        "hola",
        source_language="es",
        target_language="en",
        primary="baseten",
        fallbacks=[],
        baseten_client=baseten,
        timeout=7.0,
    )
    assert (text, leg) == ("hola-baseten", "baseten")
    assert baseten.timeouts == [7.0]


@pytest.mark.asyncio()
async def test_timeout_ignored_by_local_leg():
    engine = FakeEngine()
    text, leg = await translate_with_legs(
        "hola",
        source_language="es",
        target_language="en",
        primary="nllb",
        fallbacks=[],
        local_engine=engine,
        timeout=7.0,
    )
    assert (text, leg) == ("hola-en", "nllb")
    assert engine.calls == [("hola", "es", "en")]
