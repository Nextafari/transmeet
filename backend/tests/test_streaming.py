"""LiveStreamSession: non-blocking pump, translated partials, ordered finals.

Contract under test:
- ``handle_event`` never waits for translation (the Meta pump must keep
  draining while a translation is in flight).
- When translation is enabled, partials are translated by a throttled
  background worker (latest-wins); finals are translated in order by a
  separate worker. When disabled, partials go out raw with display debounce.
- Translation failures degrade to the source text; empty finals fall back
  to the turn's latest partial text so switched-language fragments that
  Meta does not finalize are not silently dropped.
"""

import asyncio

import pytest

from app.services.meta.streaming import LiveStreamSession
from app.services.meta.voice_client import MetaVoiceError


def _send(sent):
    async def _inner(message):
        sent.append(message)

    return _inner


def make_session(**overrides):
    sent = []

    async def translate(text, source):
        return f"{text}|{overrides.get('target', 'en')}"

    kwargs = {
        "translate_text": translate,
        "source_language": "es",
        "target_language": "en",
        "translate": True,
        "debounce_ms": 0,
        "partial_translate_ms": 0,
    }
    kwargs.update(overrides)
    translate_fn = kwargs.pop("translate_text")
    session = LiveStreamSession(translate_text=translate_fn, **kwargs)
    return session, sent


@pytest.mark.asyncio()
async def test_partial_is_translated():
    calls = []

    async def translate(text, source):
        calls.append(text)
        return f"{text}|en"

    session = LiveStreamSession(
        translate_text=translate,
        source_language="es",
        target_language="en",
        translate=True,
        debounce_ms=0,
        partial_translate_ms=0,
    )
    sent = []
    try:
        await session.handle_event(
            {"type": "transcript", "transcript": "hola", "final": False},
            _send(sent),
        )
        await session.flush(_send(sent))
        assert sent == [{"type": "partial", "text": "hola|en", "turn_id": None}]
        assert calls == ["hola"]
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_finals_translated_in_order():
    session, sent = make_session()
    try:
        await session.handle_event(
            {"type": "speechComplete", "turnId": 1, "transcript": "Hola."},
            _send(sent),
        )
        await session.handle_event(
            {"type": "speechComplete", "turnId": 2, "transcript": "Adios."},
            _send(sent),
        )
        await session.flush(_send(sent))
        assert sent == [
            {"type": "final", "text": "Hola.|en", "turn_id": 1},
            {"type": "final", "text": "Adios.|en", "turn_id": 2},
        ]
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_handle_event_never_blocks_on_translation():
    release = asyncio.Event()

    async def slow_translate(text, source):
        await release.wait()
        return f"{text}|en"

    # Finals: handle_event returns while translation is blocked.
    session = LiveStreamSession(
        translate_text=slow_translate,
        source_language="es",
        target_language="en",
        translate=True,
        debounce_ms=0,
        partial_translate_ms=0,
    )
    sent = []
    try:
        await asyncio.wait_for(
            session.handle_event(
                {"type": "speechComplete", "turnId": 1, "transcript": "Hola."},
                _send(sent),
            ),
            timeout=2,
        )
        assert sent == []
    finally:
        release.set()
        await session.aclose()

    # Partials: handle_event returns while translation is blocked.
    release.clear()
    session, sent = make_session(
        translate_text=slow_translate, partial_translate_ms=0
    )
    try:
        await asyncio.wait_for(
            session.handle_event(
                {
                    "type": "transcript",
                    "transcript": "Adios",
                    "final": False,
                    "turnId": 1,
                },
                _send(sent),
            ),
            timeout=2,
        )
        assert sent == []
        release.set()
        await session.flush(_send(sent))
        assert sent == [{"type": "partial", "text": "Adios|en", "turn_id": 1}]
    finally:
        release.set()
        await session.aclose()


@pytest.mark.asyncio()
async def test_raw_debounce_coalesces_when_translate_off():
    session, sent = make_session(translate=False, debounce_ms=60000)
    try:
        await session.handle_event(
            {"type": "transcript", "transcript": "ho", "final": False}, _send(sent)
        )
        await session.handle_event(
            {"type": "transcript", "transcript": "hola", "final": False},
            _send(sent),
        )
        assert [message["text"] for message in sent] == ["ho"]
        await session.flush(_send(sent))
        assert sent[-1] == {"type": "partial", "text": "hola", "turn_id": None}
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_translate_throttle_coalesces_rapid_partials():
    session, sent = make_session(partial_translate_ms=1000)
    try:
        await session.handle_event(
            {"type": "transcript", "transcript": "ho", "final": False}, _send(sent)
        )
        await asyncio.sleep(0.05)
        await session.handle_event(
            {"type": "transcript", "transcript": "hola", "final": False},
            _send(sent),
        )
        assert [message["text"] for message in sent] == ["ho|en"]
        await session.flush(_send(sent))
        assert sent[-1] == {"type": "partial", "text": "hola|en", "turn_id": None}
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_duplicate_partial_not_resent():
    session, sent = make_session()
    try:
        event = {"type": "transcript", "transcript": "hola", "final": False}
        await session.handle_event(event, _send(sent))
        await session.handle_event(event, _send(sent))
        await session.flush(_send(sent))
        assert sent == [{"type": "partial", "text": "hola|en", "turn_id": None}]
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_translate_false_passes_text_through_immediately():
    async def fail_translate(text, source):
        raise AssertionError("must not translate")

    session = LiveStreamSession(
        translate_text=fail_translate,
        source_language=None,
        target_language="en",
        translate=False,
        debounce_ms=0,
    )
    sent = []
    try:
        await session.handle_event(
            {"type": "transcript", "transcript": "hola", "final": True}, _send(sent)
        )
        assert sent == [{"type": "final", "text": "hola", "turn_id": None}]
        await session.handle_event(
            {"type": "transcript", "transcript": "adios", "final": False},
            _send(sent),
        )
        assert sent[-1] == {"type": "partial", "text": "adios", "turn_id": None}
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_translation_failure_degrades_to_source_text():
    async def fail_translate(text, source):
        raise RuntimeError("legs down")

    session = LiveStreamSession(
        translate_text=fail_translate,
        source_language="es",
        target_language="en",
        translate=True,
        debounce_ms=0,
        partial_translate_ms=0,
    )
    sent = []
    try:
        await session.handle_event(
            {"type": "speechComplete", "turnId": 3, "transcript": "Hola."},
            _send(sent),
        )
        await session.flush(_send(sent))
        assert sent == [{"type": "final", "text": "Hola.", "turn_id": 3}]
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_partial_translation_failure_degrades_to_source():
    async def fail_translate(text, source):
        raise RuntimeError("legs down")

    session = LiveStreamSession(
        translate_text=fail_translate,
        source_language="es",
        target_language="en",
        translate=True,
        debounce_ms=0,
        partial_translate_ms=0,
    )
    sent = []
    try:
        await session.handle_event(
            {"type": "transcript", "transcript": "hola", "final": False},
            _send(sent),
        )
        await session.flush(_send(sent))
        assert sent == [{"type": "partial", "text": "hola", "turn_id": None}]
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_empty_final_falls_back_to_pending_partial():
    session, sent = make_session(partial_translate_ms=1000)
    try:
        await session.handle_event(
            {"type": "transcript", "transcript": "ho", "final": False}, _send(sent)
        )
        await asyncio.sleep(0.05)
        await session.handle_event(
            {"type": "transcript", "transcript": "hola", "final": False},
            _send(sent),
        )
        await session.handle_event(
            {"type": "speechComplete", "turnId": 2, "transcript": ""}, _send(sent)
        )
        await session.flush(_send(sent))
        assert sent == [
            {"type": "partial", "text": "ho|en", "turn_id": None},
            {"type": "final", "text": "hola|en", "turn_id": 2},
        ]
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_empty_final_with_no_partial_sends_nothing():
    session, sent = make_session()
    try:
        await session.handle_event(
            {"type": "speechComplete", "turnId": 9, "transcript": "  "},
            _send(sent),
        )
        await session.flush(_send(sent))
        assert sent == []
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_stale_final_skipped_when_turn_advanced():
    release = asyncio.Event()

    async def gated_translate(text, source):
        if text == "Uno.":
            await release.wait()
        return f"{text}|en"

    session = LiveStreamSession(
        translate_text=gated_translate,
        source_language="es",
        target_language="en",
        translate=True,
        debounce_ms=0,
        partial_translate_ms=0,
    )
    sent = []
    try:
        await asyncio.wait_for(
            session.handle_event(
                {"type": "speechComplete", "turnId": 1, "transcript": "Uno."},
                _send(sent),
            ),
            timeout=2,
        )
        await session.handle_event(
            {"type": "transcript", "transcript": "Dos", "final": False, "turnId": 2},
            _send(sent),
        )
        await asyncio.sleep(0.1)
        assert sent == [{"type": "partial", "text": "Dos|en", "turn_id": 2}]
        release.set()
        await session.flush(_send(sent))
        # The turn-1 final finished after turn 2 was already on screen;
        # sending it would jump the overlay backwards, so it is dropped.
        assert sent == [{"type": "partial", "text": "Dos|en", "turn_id": 2}]
    finally:
        release.set()
        await session.aclose()


@pytest.mark.asyncio()
async def test_stale_partial_dropped_after_final():
    session, sent = make_session(partial_translate_ms=0)
    try:
        await session.handle_event(
            {"type": "speechComplete", "turnId": 1, "transcript": "Hola."},
            _send(sent),
        )
        await session.flush(_send(sent))
        assert sent == [{"type": "final", "text": "Hola.|en", "turn_id": 1}]
        # Late partial for an already-finalized turn must not overwrite it.
        await session.handle_event(
            {"type": "transcript", "transcript": "Hola", "final": False, "turnId": 1},
            _send(sent),
        )
        await session.flush(_send(sent))
        assert sent == [{"type": "final", "text": "Hola.|en", "turn_id": 1}]
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_duplicate_final_same_turn_sent_once():
    session, sent = make_session()
    try:
        await session.handle_event(
            {
                "type": "transcript",
                "transcript": "Hola.",
                "final": True,
                "turnId": 4,
            },
            _send(sent),
        )
        await session.handle_event(
            {"type": "speechComplete", "turnId": 4, "transcript": "Hola."},
            _send(sent),
        )
        await session.flush(_send(sent))
        assert sent == [{"type": "final", "text": "Hola.|en", "turn_id": 4}]
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_error_event_raises():
    session, _sent = make_session()
    try:
        with pytest.raises(MetaVoiceError, match="boom"):
            await session.handle_event(
                {"type": "error", "message": "boom"}, _send([])
            )
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_flush_emits_pending_partial():
    session, sent = make_session(translate=False, debounce_ms=60000)
    try:
        await session.handle_event(
            {"type": "transcript", "transcript": "ho", "final": False}, _send(sent)
        )
        await session.handle_event(
            {"type": "transcript", "transcript": "hola", "final": False},
            _send(sent),
        )
        assert len(sent) == 1
        await session.flush(_send(sent))
        assert sent[-1] == {"type": "partial", "text": "hola", "turn_id": None}
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_flush_emits_pending_translated_partial():
    session, sent = make_session(partial_translate_ms=1000)
    try:
        await session.handle_event(
            {"type": "transcript", "transcript": "ho", "final": False}, _send(sent)
        )
        await asyncio.sleep(0.05)
        await session.handle_event(
            {"type": "transcript", "transcript": "hola", "final": False},
            _send(sent),
        )
        assert len(sent) == 1
        await session.flush(_send(sent))
        assert sent[-1] == {"type": "partial", "text": "hola|en", "turn_id": None}
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_finals_backlog_coalesces_to_latest():
    calls = []

    async def translate(text, source):
        calls.append(text)
        return f"{text}|en"

    session = LiveStreamSession(
        translate_text=translate,
        source_language="es",
        target_language="en",
        translate=True,
        debounce_ms=0,
        partial_translate_ms=0,
    )
    sent = []
    try:
        for turn in (1, 2, 3, 4):
            await session.handle_event(
                {
                    "type": "speechComplete",
                    "turnId": turn,
                    "transcript": f"Turno {turn}.",
                },
                _send(sent),
            )
        await session.flush(_send(sent))
        # Four finals queued before the worker ran: the middle backlog is
        # dropped without translating so the overlay jumps to latest.
        assert calls == ["Turno 3.", "Turno 4."]
        assert sent == [
            {"type": "final", "text": "Turno 3.|en", "turn_id": 3},
            {"type": "final", "text": "Turno 4.|en", "turn_id": 4},
        ]
    finally:
        await session.aclose()


@pytest.mark.asyncio()
async def test_long_partial_translates_tail_only():
    seen = []

    async def capture(text, source):
        seen.append(text)
        return "TAIL|en"

    session = LiveStreamSession(
        translate_text=capture,
        source_language="ja",
        target_language="en",
        translate=True,
        debounce_ms=0,
        partial_translate_ms=0,
    )
    sent = []
    try:
        await session.handle_event(
            {"type": "transcript", "transcript": "あ" * 2000, "final": False},
            _send(sent),
        )
        await session.flush(_send(sent))
        assert seen == ["あ" * 600]
        assert sent == [{"type": "partial", "text": "TAIL|en", "turn_id": None}]
    finally:
        await session.aclose()
