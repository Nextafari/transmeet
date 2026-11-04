"""Local text translation engine (NLLB / Marian).

Shared by the whisper-local backend and the Meta-powered backends so the
local translation model is loaded once and reused. Heavy ML imports stay
inside ``__init__`` so importing this module never requires torch or
transformers to be installed.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Full Flores-200 codes (e.g. yor_Latn) pass straight through so callers are
# not limited to the short ISO map below; the tokenizer still rejects codes
# the loaded model does not know.
_FLORES_CODE_RE = re.compile(r"^([a-z]{3})_([a-z]{4})$")


class LocalTranslationEngine:
    """Local NLLB or Marian text translation (no network, no billing)."""

    _nllb_lang_map: dict[str, str] = {
        "en": "eng_Latn",
        "es": "spa_Latn",
        "fr": "fra_Latn",
        "de": "deu_Latn",
        "it": "ita_Latn",
        "pt": "por_Latn",
        "ru": "rus_Cyrl",
        "ja": "jpn_Jpan",
        "ko": "kor_Hang",
        "zh": "zho_Hans",
        "zh-cn": "zho_Hans",
        "zh-tw": "zho_Hant",
        "ar": "arb_Arab",
        "hi": "hin_Deva",
        "nl": "nld_Latn",
        "sv": "swe_Latn",
        "tr": "tur_Latn",
        "vi": "vie_Latn",
        "th": "tha_Thai",
        "id": "ind_Latn",
    }

    def __init__(
        self,
        provider: str = "nllb",
        model_name: str = "facebook/nllb-200-distilled-600M",
    ) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        self.provider = provider
        self.model_name = model_name
        self._torch = torch
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(
            "Initializing local translation model '%s' on %s",
            model_name,
            self._device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

    def map_language(self, lang: str | None) -> str | None:
        if not lang:
            return None
        normalized = lang.strip().lower()
        if normalized in {"auto", "detect", "auto-detect", "autodetect"}:
            return None
        mapped = self._nllb_lang_map.get(normalized)
        if mapped is not None:
            return mapped
        match = _FLORES_CODE_RE.match(normalized)
        if match:
            return f"{match.group(1)}_{match.group(2).capitalize()}"
        return None

    def same_language(
        self,
        source_language: str | None,
        target_language: str,
    ) -> bool:
        source_nllb = self.map_language(source_language)
        target_nllb = self.map_language(target_language)
        return bool(source_nllb and target_nllb and source_nllb == target_nllb)

    def is_same_language(
        self,
        source_language: str | None,
        target_language: str,
    ) -> bool:
        """Provider-aware same-language check (no model needed).

        Single-pair engines (marian) compare plain codes since NLLB code
        mapping does not apply to them.
        """
        if self.provider == "marian":
            source_code = str(source_language or "").strip().lower()
            target_code = str(target_language or "").strip().lower()
            return bool(source_code and source_code == target_code)
        return self.same_language(source_language, target_language)

    def supports_source(self, source_language: str | None) -> bool:
        if self.provider == "marian":
            return True
        return self.map_language(source_language) is not None

    def translate(
        self,
        text: str,
        source_language: str | None,
        target_language: str,
    ) -> str:
        if self.provider == "marian":
            return self._translate_marian(text)
        return self._translate_nllb(text, source_language, target_language)

    def _translate_nllb(
        self,
        text: str,
        source_language: str | None,
        target_language: str,
    ) -> str:
        def _get_lang_id(lang_code: str) -> int | None:
            if hasattr(self._tokenizer, "lang_code_to_id"):
                lang_map = getattr(self._tokenizer, "lang_code_to_id") or {}
                return lang_map.get(lang_code)
            if hasattr(self._tokenizer, "get_lang_id"):
                return self._tokenizer.get_lang_id(lang_code)
            if hasattr(self._tokenizer, "convert_tokens_to_ids"):
                return self._tokenizer.convert_tokens_to_ids(lang_code)
            return None

        src_lang = self.map_language(source_language)
        tgt_lang = self.map_language(target_language)
        if src_lang is None:
            raise RuntimeError(
                f"NLLB source language is unsupported or unknown: {source_language}"
            )
        if tgt_lang is None:
            raise RuntimeError(
                f"NLLB target language is unsupported or unknown: {target_language}"
            )

        self._tokenizer.src_lang = src_lang
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        forced_bos_token_id = _get_lang_id(tgt_lang)
        if forced_bos_token_id is None:
            raise RuntimeError(
                f"Unable to resolve target language token id for NLLB: {tgt_lang}"
            )

        input_len = int(encoded["input_ids"].shape[1])
        with self._torch.inference_mode():
            output = self._model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=min(128, max(32, input_len + 16)),
                num_beams=1,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        return self._tokenizer.batch_decode(
            output, skip_special_tokens=True
        )[0]

    def _translate_marian(self, text: str) -> str:
        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        input_len = int(encoded["input_ids"].shape[1])
        with self._torch.inference_mode():
            output = self._model.generate(
                **encoded,
                max_new_tokens=min(128, max(32, input_len + 16)),
                num_beams=1,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        return self._tokenizer.batch_decode(
            output, skip_special_tokens=True
        )[0]
