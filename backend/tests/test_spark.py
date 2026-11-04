"""Muse Spark client: payload shape, reused connection, per-call timeout."""

import json

import httpx
import pytest

from app.services.meta.spark_client import MetaSparkClient, MetaSparkError


def _ok_payload(text="Hola."):
    return {
        "choices": [
            {"message": {"content": json.dumps({"translated_text": text})}}
        ]
    }


def _make_client(handler, **kwargs):
    return MetaSparkClient(
        api_key="key", transport=httpx.MockTransport(handler), **kwargs
    )


@pytest.mark.asyncio()
async def test_translate_posts_chat_completions_and_extracts():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_ok_payload("Hola."))

    client = _make_client(handler)
    try:
        assert await client.translate_text("Hello.", "es") == "Hola."
    finally:
        await client.aclose()
    assert seen["url"] == "https://api.meta.ai/v1/chat/completions"
    assert seen["auth"] == "Bearer key"
    assert seen["body"]["model"] == "muse-spark-1.3"
    # Generous cap: the model reasons before answering, and reasoning
    # counts against max_tokens; a tight cap truncates to empty content.
    assert seen["body"]["max_tokens"] == 2048
    user_content = seen["body"]["messages"][1]["content"]
    assert "Target language: es" in user_content
    assert "Hello." in user_content


@pytest.mark.asyncio()
async def test_length_truncation_named() -> None:
    payload = {
        "choices": [
            {"message": {"content": None}, "finish_reason": "length"}
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = _make_client(handler)
    try:
        with pytest.raises(MetaSparkError, match="truncated"):
            await client.translate_text("Hello.", "es")
    finally:
        await client.aclose()


@pytest.mark.asyncio()
async def test_client_reused_across_calls():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_ok_payload("x"))

    client = _make_client(handler)
    try:
        assert client._get_client() is client._get_client()
        await client.translate_text("a", "es")
        await client.translate_text("b", "es")
        assert len(calls) == 2
    finally:
        await client.aclose()


@pytest.mark.asyncio()
async def test_per_call_timeout_override():
    extension_timeouts = []

    def handler(request: httpx.Request) -> httpx.Response:
        extension_timeouts.append(request.extensions.get("timeout"))
        return httpx.Response(200, json=_ok_payload("x"))

    client = _make_client(handler)
    try:
        await client.translate_text("a", "es")
        await client.translate_text("b", "es", timeout=7.0)
    finally:
        await client.aclose()
    assert extension_timeouts[0]["read"] == 30.0
    assert extension_timeouts[1]["read"] == 7.0


@pytest.mark.asyncio()
async def test_rate_limit_marks_quota():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    client = _make_client(handler)
    try:
        with pytest.raises(MetaSparkError) as exc_info:
            await client.translate_text("a", "es")
    finally:
        await client.aclose()
    assert exc_info.value.quota is True


@pytest.mark.asyncio()
async def test_list_content_blocks_joined():
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": '{"translated_text": "Ho'},
                        {"type": "text", "text": 'la"}'},
                    ]
                }
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = _make_client(handler)
    try:
        assert await client.translate_text("Hello.", "es") == "Hola"
    finally:
        await client.aclose()


@pytest.mark.asyncio()
async def test_refusal_surfaces_reason():
    payload = {
        "choices": [
            {"message": {"content": None, "refusal": "policy block"}, "finish_reason": "stop"}
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = _make_client(handler)
    try:
        with pytest.raises(MetaSparkError, match="policy block"):
            await client.translate_text("Hello.", "es")
    finally:
        await client.aclose()


@pytest.mark.asyncio()
async def test_unusable_content_logs_shape(caplog):
    payload = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = _make_client(handler)
    try:
        with pytest.raises(MetaSparkError, match="empty content"):
            await client.translate_text("Hello.", "es")
    finally:
        await client.aclose()
    assert "choices=1" in caplog.text
    assert "finish_reason=stop" in caplog.text
    assert "content_type=NoneType" in caplog.text
