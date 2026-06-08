# Copyright (c) 2026 Pengshen Zhang
"""Transcript Postprocess: ASR 文本清洗与语言元信息提取。

- strip_asr_tags: 去掉模型输出中的 <asr_text> 标签
- split_language_prefix: 提取 language 前缀并返回正文
- postprocess_transcript: 统一执行空白、标点、语言字段规范化
- 保持纯字符串处理，不依赖 engine/session 状态
"""
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple

_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "

_TRAILING_PUNCT = frozenset(
    "，。,.!?！？、；;：:…·—–"
)


def _is_trailing_punct(ch: str) -> bool:
    if ch in _TRAILING_PUNCT:
        return True
    return unicodedata.category(ch).startswith("P")


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
