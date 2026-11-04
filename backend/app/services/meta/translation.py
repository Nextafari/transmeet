"""Translation-leg resolution for API-powered backends.

Legs: ``nllb`` (local engine: free, offline), ``spark`` (Muse Spark via
the Meta Model API: billed), and ``baseten`` (hosted open model via
Baseten Model APIs: billed). Order is server-side only, from settings:
``META_TRANSLATION_ENGINE`` first, then ``STREAM_TRANSLATION_FALLBACKS``.
Unknown leg names fail fast because they are configuration errors.
"""

import asyncio
import logging
import re

logger = logging.getLogger(__name__)

KNOWN_LEGS = ("nllb", "spark", "baseten")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")
_FRAGMENT_SPLIT_RE = re.compile(r",\s*")

# Runs of this many identical consecutive phrases/sentences collapse to one.
# Shorter runs (emphasis, stutter) are left alone.
_REPETITION_RUN = 4


def _repetition_key(unit: str) -> str:
    key = unit.strip().lower()
    if key.startswith("and "):
        key = key[4:]
    return key


def _collapse_units(units: list[str]) -> tuple[list[str], bool]:
    collapsed: list[str] = []
    changed = False
    index = 0
    while index < len(units):
        end = index + 1
        while (
            end < len(units)
            and _repetition_key(units[end]) == _repetition_key(units[index])
        ):
            end += 1
        run = units[index:end]
        if len(run) >= _REPETITION_RUN:
            collapsed.append(run[0])
            changed = True
        else:
            collapsed.extend(run)
        index = end
    return collapsed, changed


def collapse_repetition(text: str) -> str:
    """Collapse 4+ identical consecutive phrases/sentences to one.

    Guards captions against translation-loop floods
    ("X, and X, and X, ...") while leaving short emphasis untouched.
    Returns the input unchanged when nothing collapses.
    """
    if not text:
        return text
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return text
    changed_any = False
    rebuilt: list[str] = []
    for sentence in sentences:
        fragments = [f for f in _FRAGMENT_SPLIT_RE.split(sentence) if f.strip()]
        if len(fragments) > 1:
            fragments, changed = _collapse_units(fragments)
            changed_any = changed_any or changed
            rebuilt.append(", ".join(fragments))
        else:
            rebuilt.append(sentence.strip())
    rebuilt, changed = _collapse_units(rebuilt)
    changed_any = changed_any or changed
    if not changed_any:
        return text
    return " ".join(rebuilt)


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
            "to one of ('nllb', 'spark', 'baseten')."
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
    baseten_client: object | None = None,
    timeout: float | None = None,
) -> tuple[str, str]:
    """Translate text via the first working leg.

    Returns ``(translated_text, leg_used)``. Raises the last error when
    every configured leg fails. ``timeout`` overrides the per-request
    network timeout for API legs; the local leg ignores it.
    """
    legs = ordered_legs(primary, fallbacks)
    if not (text or "").strip():
        return "", legs[0]

    # Same-language shortcut (mirrors the whisper-local backends): no call
    # is needed when source and target are the same language.
    engine = local_engine
    if engine is not None:
        try:
            if engine.is_same_language(source_language, target_language):  # type: ignore[attr-defined]
                logger.info(
                    "Translation leg skipped: source and target are both %s",
                    target_language,
                )
                return text, legs[0]
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
                return collapse_repetition(translated), "nllb"
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
                    timeout=timeout,
                )
                return collapse_repetition(translated), "spark"
            if leg == "baseten":
                if baseten_client is None:
                    raise RuntimeError(
                        "The 'baseten' leg needs BASETEN_API_KEY, which is "
                        "not configured."
                    )
                translated = await baseten_client.translate_text(  # type: ignore[attr-defined]
                    text,
                    target_language,
                    source_language=source_language,
                    timeout=timeout,
                )
                return collapse_repetition(translated), "baseten"
        except Exception as exc:
            last_error = exc if isinstance(exc, Exception) else RuntimeError(str(exc))
            logger.warning("Translation leg '%s' failed: %s", leg, exc)

    assert last_error is not None
    raise last_error
