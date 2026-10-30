"""Translation-leg resolution for Meta-powered backends.

Legs: ``nllb`` (local engine: free, offline) and ``spark`` (Muse Spark via
the Meta Model API: billed). Order is server-side only, from settings:
``META_TRANSLATION_ENGINE`` first, then ``STREAM_TRANSLATION_FALLBACKS``.
Unknown leg names fail fast because they are configuration errors.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

KNOWN_LEGS = ("nllb", "spark")


def ordered_legs(
    primary: str | None,
    fallbacks: list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Deduplicated leg order. Raises ValueError on unknown leg names."""
    legs: list[str] = []
    for name in [primary, *(fallbacks or [])]:
        normalized = str(name or "").strip().lower()
        if not normalized or normalized in legs:
            continue
        if normalized not in KNOWN_LEGS:
            raise ValueError(
                f"Unknown translation leg: {name!r}. Use one of {KNOWN_LEGS}."
            )
        legs.append(normalized)
    if not legs:
        raise ValueError(
            "No translation leg configured. Set META_TRANSLATION_ENGINE "
            "to one of ('nllb', 'spark')."
        )
    return legs


async def translate_with_legs(
    text: str,
    *,
    source_language: str | None,
    target_language: str,
    primary: str | None,
    fallbacks: list[str] | tuple[str, ...] | None,
    local_engine: object | None = None,
    spark_client: object | None = None,
) -> tuple[str, str]:
    """Translate text via the first working leg.

    Returns ``(translated_text, leg_used)``. Raises the last error when
    every configured leg fails.
    """
    legs = ordered_legs(primary, fallbacks)
    if not (text or "").strip():
        return "", legs[0]

    # Same-language shortcut (mirrors the whisper-local backends): no call
    # is needed when source and target resolve to the same NLLB language.
    engine = local_engine
    if engine is not None and getattr(engine, "provider", "nllb") == "nllb":
        try:
            if engine.same_language(source_language, target_language):  # type: ignore[attr-defined]
                logger.info(
                    "Translation leg skipped: source and target are both %s",
                    target_language,
                )
                return text, "nllb"
        except Exception:
            pass

    last_error: Exception | None = None
    for leg in legs:
        try:
            if leg == "nllb":
                if engine is None:
                    raise RuntimeError(
                        "The 'nllb' leg needs a local translation engine, "
                        "which is unavailable (torch/transformers missing?)."
                    )
                translated = await asyncio.to_thread(
                    engine.translate,  # type: ignore[attr-defined]
                    text,
                    source_language,
                    target_language,
                )
                return translated, "nllb"
            if leg == "spark":
                if spark_client is None:
                    raise RuntimeError(
                        "The 'spark' leg needs META_MODEL_API_KEY, which is "
                        "not configured."
                    )
                translated = await spark_client.translate_text(  # type: ignore[attr-defined]
                    text,
                    target_language,
                    source_language=source_language,
                )
                return translated, "spark"
        except Exception as exc:
            last_error = exc if isinstance(exc, Exception) else RuntimeError(str(exc))
            logger.warning("Translation leg '%s' failed: %s", leg, exc)

    assert last_error is not None
    raise last_error
