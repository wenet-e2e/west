# Copyright (c) 2026 Pengshen Zhang
"""History Rollback: 历史前缀回退策略。

- HistoryRollbackConfig: none/ratio/chars/words/tokens 策略配置
- compute_rollback_chars: 计算需要从历史尾部回退的字符数
- apply_history_rollback_result: 返回回退后的前缀和诊断信息
- 支持中文 jieba 分词和 tokenizer 级回退，供流式纠错使用
"""
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import jieba as _jieba

logger = logging.getLogger("RealtimeASR")

_jieba.setLogLevel(logging.WARNING)

# CJK Unified Ideographs + Extension A. Used to route Chinese text to
# jieba word rollback.
_CJK_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
RollbackStrategy = Literal["none", "ratio", "chars", "words", "tokens"]


def _has_cjk(text: str) -> bool:
    return bool(_CJK_PATTERN.search(text))


def _rollback_words_by_space(text: str, n_words: int) -> int:
    """英文：按空格从尾部回退 N 个词，返回要删掉的字符数。"""
    if n_words <= 0:
        return 0
    matches = list(re.finditer(r"\S+", text))
    if n_words >= len(matches):
        return len(text)
    rollback_start = matches[-n_words].start()
    return len(text) - rollback_start


def _rollback_words_by_jieba(text: str, n_words: int) -> int:
    """中文：按 jieba 分词从尾部回退 N 个词，返回要删掉的字符数。"""
    segs = list(_jieba.cut(text))
    if n_words >= len(segs):
        return len(text)
    kept = "".join(segs[:-n_words])
    return len(text) - len(kept)


def _rollback_tokens(text: str, n_tokens: int, tokenizer) -> int:
    """Token 级回退：从尾部回退 N 个 token，返回要删掉的字符数。

    采用 Qwen3-ASR 的策略：逐步增加回退量直到解码后的文本不含 Unicode 错误字符。

    Args:
        text: 要回退的文本
        n_tokens: 要回退的 token 数
        tokenizer: Transformers tokenizer（需要支持 encode/decode）

    Returns:
        int: 要删掉的字符数
    """
    if not text or n_tokens <= 0:
        return 0

    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if n_tokens >= len(token_ids):
            return len(text)

        # 回退 n_tokens 个 token，处理 Unicode 不完整字符
        k = n_tokens
        while True:
            end_idx = max(0, len(token_ids) - k)
            if end_idx == 0:
                return len(text)  # 全部回退

            kept = tokenizer.decode(token_ids[:end_idx])

            # 检查是否有 Unicode 替换字符（表示不完整的字符）
            if '\ufffd' not in kept:
                return len(text) - len(kept)

            # 有不完整字符，增加回退量重试
            k += 1
            if k >= len(token_ids):
                return len(text)  # 避免无限循环
    except Exception as e:
        tokenizer_type = type(
            tokenizer).__name__ if tokenizer is not None else "None"
        logger.warning(
            "Token rollback failed, falling back to char-based rollback "
            f"(n_tokens={n_tokens}, tokenizer={tokenizer_type}): {e}")
        # 降级到字符级回退
        return min(n_tokens, len(text))


@dataclass(frozen=True)
class HistoryRollbackConfig:
    strategy: RollbackStrategy = "none"
    value: float = 0.0

    VALID_STRATEGIES = ("none", "ratio", "chars", "words", "tokens")

    def __post_init__(self):
        strategy = self.strategy
        if strategy not in self.VALID_STRATEGIES:
            strategy = "none"
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "value", float(self.value))

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'HistoryRollbackConfig':
        if not d or not d.get("enabled", False):
            return cls("none", 0.0)
        return cls(
            strategy=d.get("strategy", "none"),
            value=float(d.get("value", 0.0)),
        )

    def compute_rollback_chars(self, text: str, tokenizer=None) -> int:
        """计算要回退的字符数。

        Args:
            text: 要回退的文本
            tokenizer: 可选的 tokenizer（仅 tokens 策略需要）

        Returns:
            int: 要删掉的字符数
        """
        if not text or self.strategy == "none" or self.value <= 0:
            return 0
        if self.strategy == "ratio":
            return max(0, int(len(text) * min(self.value, 1.0)))
        elif self.strategy == "chars":
            return min(int(self.value), len(text))
        elif self.strategy == "words":
            n_words = int(self.value)
            if _has_cjk(text):
                return _rollback_words_by_jieba(text, n_words)
            return _rollback_words_by_space(text, n_words)
        elif self.strategy == "tokens":
            n_tokens = int(self.value)
            if tokenizer is None:
                logger.warning(
                    "tokens strategy requires tokenizer, "
                    f"falling back to char-based rollback (n={n_tokens})")
                return min(n_tokens, len(text))
            return _rollback_tokens(text, n_tokens, tokenizer)
        return 0

    def apply(self, text: str, tokenizer=None) -> str:
        """应用回退策略到文本。

        Args:
            text: 要回退的文本
            tokenizer: 可选的 tokenizer（仅 tokens 策略需要）

        Returns:
            str: 回退后的文本
        """
        rb = self.compute_rollback_chars(text, tokenizer=tokenizer)
        if rb <= 0:
            return text
        return text[:-rb]


@dataclass(frozen=True)
class HistoryRollbackResult:
    text: str
    dropped: str
    reason: str
    rollback_chars: int
    strategy: str
    value: float


def apply_history_rollback_result(
        history: str,
        history_rollback_config: Optional['HistoryRollbackConfig'] = None,
        tokenizer=None,
        current_chunk_id: int = 0,
        history_reset_chunk_num: int = 0,
        min_history_chars: int = 0) -> HistoryRollbackResult:
    """返回历史前缀和诊断信息。"""
    cfg = history_rollback_config or HistoryRollbackConfig()
    strategy = cfg.strategy
    value = cfg.value

    if not history:
        return HistoryRollbackResult(
            "", "", "empty_history", 0, strategy, value)

    if current_chunk_id <= history_reset_chunk_num:
        return HistoryRollbackResult(
            "", history, "history_reset", len(history), strategy, value)

    if min_history_chars > 0 and len(history) <= min_history_chars:
        return HistoryRollbackResult(
            "", history, "min_history_chars", len(history), strategy, value)

    if strategy == "none":
        return HistoryRollbackResult(history, "", "none", 0, strategy, value)

    rollback_chars = cfg.compute_rollback_chars(history, tokenizer=tokenizer)
    if rollback_chars <= 0:
        return HistoryRollbackResult(
            history, "", "history_rollback_noop", 0, strategy, value)

    kept = history[:-rollback_chars]
    dropped = history[-rollback_chars:]
    return HistoryRollbackResult(
        kept, dropped, "history_rollback", rollback_chars, strategy, value)


def apply_history_rollback(
        history: str,
        history_rollback_config: Optional['HistoryRollbackConfig'] = None,
        tokenizer=None,
        current_chunk_id: int = 0,
        history_reset_chunk_num: int = 0,
        min_history_chars: int = 0) -> str:
    """返回要拼接到 prompt 的历史文本。

    无回退: 拼接完整 history
    有回退: 去掉尾部不确定部分，只拼确定的前缀
    前 N 块全解: 前 N 块不使用任何历史前缀（完全重新推理）
    短历史保护: history 字符数 <= min_history_chars 时清空（避免模型 EOS）

    两个保护策略为 OR 关系，任一触发则返回空字符串。

    Args:
        history: 历史文本
        history_rollback_config: 回退配置
        tokenizer: 可选的 tokenizer（仅 tokens 策略需要）
        current_chunk_id: 当前 chunk ID（从 1 开始，extract 后自增）
        history_reset_chunk_num: 前 N 块不使用历史前缀
        min_history_chars: 历史文本最小字符数，<=此值则丢弃

    Returns:
        str: 回退后的历史文本（保护策略触发时返回空字符串）
    """
    return apply_history_rollback_result(
        history,
        history_rollback_config,
        tokenizer=tokenizer,
        current_chunk_id=current_chunk_id,
        history_reset_chunk_num=history_reset_chunk_num,
        min_history_chars=min_history_chars,
    ).text
