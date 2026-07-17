# Copyright (c) 2026 Pengshen Zhang
"""Transcript Postprocess: ASR 文本清洗与语言元信息提取。

- strip_asr_tags: 去掉模型输出中的 <asr_text> 标签
- split_language_prefix: 提取 language 前缀并返回正文
- postprocess_transcript: 统一执行空白、标点、语言字段规范化
- 保持纯字符串处理，不依赖 engine/session 状态
"""
import logging
import tempfile
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("RealtimeASR")

_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "

_TRAILING_PUNCT = frozenset(
    "，。,.!?！？、；;：:…·—–"
)
_LEADING_PUNCT = frozenset(
    "([{《〈「『“‘"
)
_ITN_UNAVAILABLE_ERRORS = {}
_WETEXT_CACHE_DIR = (
    Path(tempfile.gettempdir()) / "serve_touch_asr" / "wetextprocessing" /
    "itn")
_WETEXT_NORMALIZERS = {
    "zh": "itn.chinese.inverse_normalizer",
}


def _is_latin_word_char(ch: str) -> bool:
    """ASCII alphanumeric or common word-internal punctuation."""
    return ch.isascii() and (ch.isalnum() or ch in "_'-")


def _is_cjk_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _is_trailing_punct(ch: str) -> bool:
    if ch in _TRAILING_PUNCT:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_leading_punct(ch: str) -> bool:
    return ch in _LEADING_PUNCT or unicodedata.category(ch).startswith("P")


def _is_latin_letter(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def merge_transcript_boundary(left: str, right: str) -> str:
    """Merge adjacent transcript segments without gluing boundaries.

    Rules are matched top-down to decide whether to insert a space:
      1. Either side is whitespace, or right starts with leading punct
         (open bracket, etc.)            -> join as-is
      2. Latin word + Latin word         -> space ("hello"+"world")
      3. Latin word <-> CJK              -> space ("hello"+"世界")
      4. Word-internal punct (' -) + Latin letter
                                         -> join as-is (chunk split mid-word)
      5. Punct + Latin letter            -> space ("hello,"+"world")
      6. Half-width punct + CJK          -> space ("day."+"今天"); CJK punct
         (。，) before CJK keeps no space, per CJK typography
      7. Otherwise                       -> join as-is
    """
    if not left:
        return right
    if not right:
        return left

    lch, rch = left[-1], right[0]
    if lch.isspace() or rch.isspace() or _is_leading_punct(rch):
        return left + right
    if _is_latin_word_char(lch) and _is_latin_word_char(rch):
        return f"{left} {right}"
    if (_is_latin_word_char(lch) and _is_cjk_char(rch)
            or _is_cjk_char(lch) and _is_latin_word_char(rch)):
        return f"{left} {right}"
    if lch in "'-" and _is_latin_letter(rch):
        return left + right
    if _is_trailing_punct(lch) and _is_latin_letter(rch):
        return f"{left} {right}"
    if lch.isascii() and _is_trailing_punct(lch) and _is_cjk_char(rch):
        return f"{left} {right}"
    return left + right


def _itn_lang(language: Optional[str], text: str) -> str:
    """Pick the ITN normalizer language.

    Only Chinese ITN is supported. English and other explicit non-Chinese
    languages remain raw even when ITN is enabled. Unknown language falls back
    to Chinese ITN only when the transcript contains CJK characters.
    """
    lang = (language or "").strip().lower()
    if lang.startswith("en") or lang.startswith("english"):
        return ""
    if (
        lang.startswith("zh")
        or lang.startswith("chinese")
        or lang.startswith("mandarin")
        or lang.startswith("cantonese")
        or lang.startswith("yue")
    ):
        return "zh"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "zh"
    return ""


@lru_cache(maxsize=4)
def _get_inverse_normalizer(lang: str):
    module_name = _WETEXT_NORMALIZERS.get(lang)
    if module_name is None:
        raise RuntimeError(f"WeTextProcessing ITN unsupported language: {lang}")

    module = import_module(module_name)

    cache_dir = _WETEXT_CACHE_DIR / lang
    cache_dir.mkdir(parents=True, exist_ok=True)
    return module.InverseNormalizer(
        cache_dir=str(cache_dir),
        overwrite_cache=False,
    )


def warmup_inverse_normalizer(
    language: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """Preload the Chinese ITN normalizer.

    Returns:
        (available, lang, error_message)
    """
    del language
    lang = "zh"
    if lang in _ITN_UNAVAILABLE_ERRORS:
        return False, lang, _ITN_UNAVAILABLE_ERRORS[lang]
    try:
        _get_inverse_normalizer(lang)
        return True, lang, ""
    except Exception as exc:
        error = str(exc)
        _ITN_UNAVAILABLE_ERRORS[lang] = error
        return False, lang, error


def inverse_normalize_transcript(
    text: str,
    language: Optional[str] = None,
    enabled: bool = False,
) -> str:
    """Optionally convert spoken-form ASR text to written form."""
    if not enabled or not text:
        return text

    lang = _itn_lang(language, text)
    if not lang:
        return text
    if lang in _ITN_UNAVAILABLE_ERRORS:
        return text

    try:
        return _get_inverse_normalizer(lang).normalize(text)
    except Exception as exc:
        _ITN_UNAVAILABLE_ERRORS[lang] = str(exc)
        logger.warning(
            "ITN failed; falling back to raw transcript "
            f"(lang={lang}, text={text[:80]!r}): {exc}")
        return text


def _strip_trailing_punct(text: str) -> Tuple[str, str]:
    """Strip one trailing punctuation character.

    Returns:
        (text_without_trailing, stripped_char)
    """
    if not text:
        return text, ""
    last = text[-1]
    if _is_trailing_punct(last):
        return text[:-1], last
    return text, ""


# ── qwen3-asr ────────────────────────────────────────────

def _postprocess_qwen3_asr(
    raw_text: str,
    user_language: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Parse and clean qwen3-asr output.

    Returns:
        (language, text, trailing_punct)
    """
    if not raw_text:
        return "", "", ""

    s = raw_text.strip()
    if not s:
        return "", "", ""

    if user_language:
        text, trailing = _strip_trailing_punct(s)
        return user_language, text, trailing

    if _ASR_TEXT_TAG in s:
        meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
        text_part = text_part.strip()

        if "language none" in meta_part.lower() and not text_part:
            return "", "", ""

        language = ""
        for line in meta_part.splitlines():
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith(_LANG_PREFIX):
                val = line[len(_LANG_PREFIX):].strip()
                if val:
                    language = val[:1].upper() + val[1:].lower()
                break

        text, trailing = _strip_trailing_punct(text_part)
        return language, text, trailing

    # No metadata tag — treat as plain text (streaming intermediate state
    # where the model has not yet emitted <asr_text>)
    if s.lower().startswith("language"):
        return "", "", ""

    text, trailing = _strip_trailing_punct(s)
    return "", text, trailing


# ── qwen3-omni ───────────────────────────────────────────

def _postprocess_qwen3_omni(
    raw_text: str,
) -> Tuple[str, str, str]:
    """Clean qwen3-omni output.

    Returns:
        ("", text, trailing_punct)
    """
    if not raw_text:
        return "", "", ""

    text = raw_text.strip()
    if not text:
        return "", "", ""

    text, trailing = _strip_trailing_punct(text)
    return "", text, trailing


# ── History prefix formatting (inverse of transcript_postprocess) ───

def format_history_prefix(
    text: str,
    language: str,
    model_type: str,
) -> str:
    """Wrap plain text into the history prefix format appended to prompt.

    qwen3-asr:  "language Chinese <asr_text>文本内容"
    qwen3-omni: "文本内容" (returned as-is)
    """
    if not text:
        return ""
    if model_type == "qwen3-asr":
        return f"language {language} {_ASR_TEXT_TAG}{text}"
    return text


# ── Public API ────────────────────────────────────────────

@dataclass
class TranscriptPostprocessResult:
    language: str
    text: str
    trailing_punct: str


def postprocess_transcript(
    raw_text: str,
    model_type: str,
    user_language: Optional[str] = None,
) -> TranscriptPostprocessResult:
    """Unified post-processing for ASR model outputs.

    Always strips trailing punctuation. Caller decides whether to
    restore it (e.g. at commit time) via ``result.trailing_punct``.

    Args:
        raw_text: Raw model output (accumulated tokens).
        model_type: ``"qwen3-asr"`` or ``"qwen3-omni"``.
        user_language: Forced language (qwen3-asr only).

    Returns:
        TranscriptPostprocessResult with language, clean text, and the
        stripped trailing punctuation character.
    """
    if model_type == "qwen3-asr":
        lang, text, trailing = _postprocess_qwen3_asr(
            raw_text, user_language=user_language)
    elif model_type == "qwen3-omni":
        lang, text, trailing = _postprocess_qwen3_omni(raw_text)
    else:
        text = (raw_text or "").strip()
        text, trailing = _strip_trailing_punct(text)
        lang = ""

    return TranscriptPostprocessResult(
        language=lang, text=text, trailing_punct=trailing)
