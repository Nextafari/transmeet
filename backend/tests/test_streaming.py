"""LiveStreamSession: partials, debounce, finals, degradation."""

import pytest

from app.services.meta.streaming import LiveStreamSession
from app.services.meta.voice_client import MetaVoiceError


def make_session(**overrides):
    sent = []

    async def send(message):
        sent.append(message)

    async def translate(text, source):
        return f"{text}|{overrides.get('target', 'en')}"

    kwargs = {
        "translate_text": translate,
        "source_language": "es",
        "target_language": "en",
        "translate": True,
        "debounce_ms": 0,
    }
    kwargs.update(overrides)
    translate_fn = kwargs.pop("translate_text")
    session = LiveStreamSession(translate_text=translate_fn, **kwargs)
    return session, sent


@pytest.mark.asyncio()
async def test_partial_then_final_translated():
    session, sent = make_session()
    await session.handle_event(
        {"type": "transcript", "transcript": "hola", "final": False}, _send(sent)
    )
    await session.handle_event(
        {"type": "speechComplete", "turnId": 1, "transcript": "Hola."}, _send(sent)
    )
    assert sent[0] == {"type": "partial", "text": "hola|en", "turn_id": None}
    assert sent[1] == {"type": "final", "text": "Hola.|en", "turn_id": 1}


def _send(sent):
    async def _inner(message):
        sent.append(message)

    return _inner


@pytest.mark.asyncio()
async def test_debounce_coalesces_rapid_partials():
    session, sent = make_session(debounce_ms=60000)
    await session.handle_event(
        {"type": "transcript", "transcript": "ho", "final": False}, _send(sent)
    )
    await session.handle_event(
        {"type": "transcript", "transcript": "hola", "final": False}, _send(sent)
    )
    assert [message["text"] for message in sent] == ["ho|en"]
    await session.handle_event(
        {"type": "speechComplete", "turnId": 2, "transcript": "Hola."}, _send(sent)
    )
    assert sent[-1] == {"type": "final", "text": "Hola.|en", "turn_id": 2}


@pytest.mark.asyncio()
async def test_duplicate_partial_not_resent():
    session, sent = make_session()
    event = {"type": "transcript", "transcript": "hola", "final": False}
    await session.handle_event(event, _send(sent))
    await session.handle_event(event, _send(sent))
    assert len(sent) == 1


@pytest.mark.asyncio()
async def test_translate_false_passes_text_through():
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
    await session.handle_event(
        {"type": "transcript", "transcript": "hola", "final": True}, _send(sent)
    )
    assert sent == [{"type": "final", "text": "hola", "turn_id": None}]


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
    )
    sent = []
    await session.handle_event(
        {"type": "speechComplete", "turnId": 3, "transcript": "Hola."}, _send(sent)
    )
    assert sent == [{"type": "final", "text": "Hola.", "turn_id": 3}]


@pytest.mark.asyncio()
async def test_error_event_raises():
    session, _sent = make_session()
    with pytest.raises(MetaVoiceError, match="boom"):
        await session.handle_event({"type": "error", "message": "boom"}, _send([]))


@pytest.mark.asyncio()
async def test_flush_emits_pending_partial():
    session, sent = make_session(debounce_ms=60000)
    await session.handle_event(
        {"type": "transcript", "transcript": "ho", "final": False}, _send(sent)
    )
    await session.handle_event(
        {"type": "transcript", "transcript": "hola", "final": False}, _send(sent)
    )
    assert len(sent) == 1
    await session.flush(_send(sent))
    assert sent[-1] == {"type": "partial", "text": "hola|en", "turn_id": None}
