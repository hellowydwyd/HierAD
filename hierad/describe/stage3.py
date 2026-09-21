"""
Stage 3: LLM 精炼 AD 脚本（单段、无跨段记忆）

只依据本段视觉事实 + 字数预算；不读近期 AD / 对白 / 全局剧情。
"""

from __future__ import annotations

import re
from typing import List, Optional

from openai import OpenAI

from hierad.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from hierad.progress import ProgressReporter, progress_iter

from .models import ADSegment, EnhancedGlobalContext
from .prompts import STAGE3_SYSTEM, STAGE3_USER


def max_words_for_gap(gap_sec: float, *, words_per_sec: float = 2.2) -> int:
    """按静音间隙估算可说完的词数（偏保守，配合 TTS 加速）。"""
    if gap_sec <= 0:
        return 6
    return max(5, min(22, int(gap_sec * words_per_sec)))


def _characters_are_none(chars: str) -> bool:
    c = (chars or "").strip().lower()
    return (not c) or c in {"none", "n/a", "na", "null", "-", "unknown"}


def _extract_characters_from_visual(visual: str) -> str:
    if not visual:
        return ""
    m = re.search(
        r"(?:\*\*)?Characters(?:\*\*)?\s*:\s*(.+?)(?:\n|$)",
        visual,
        flags=re.I,
    )
    if not m:
        return ""
    return m.group(1).strip().strip("*").strip()


def _resolve_characters(seg: ADSegment) -> str:
    """优先用字段；为空则从 visual markdown 回填。"""
    chars = (seg.characters or "").strip()
    if chars:
        return chars
    return _extract_characters_from_visual(seg.visual_description or "")


def _should_forbid_names(seg: ADSegment) -> bool:
    """仅当 Characters 明确为 None 时禁止专名；字段缺失且无标记时不 scrub。"""
    resolved = _resolve_characters(seg)
    if resolved:
        return _characters_are_none(resolved)
    return False


def _fallback_from_visual(visual: str, max_words: int, no_names: bool) -> str:
    """无 LLM 时从 visual 抽一句短描述。"""
    text = visual.strip()
    for key in ("**Action:**", "ACTION:", "Action:"):
        if key in text:
            text = text.split(key, 1)[-1].strip()
            break
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if no_names:
        text = re.sub(
            r"\b(Tony Stark|Obadiah Stane|Christine Everhart|Christine Everheart|"
            r"Tony|Rhodey|Rhodes|Obadiah|Christine|Hogan|Jimmy|Raza|Everhart|Everheart|Stane)\b",
            " ",
            text,
            flags=re.I,
        )
        text = re.sub(r"\s+", " ", text).strip(" ,.;:")
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        if not text or len(text.split()) < 3:
            m = re.search(
                r"(?:\*\*)?Setting(?:\*\*)?\s*:\s*(.+?)(?:\n|$)",
                visual,
                flags=re.I,
            )
            if m:
                text = m.group(1).strip().strip("*").strip()
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;:") + "."
    return text


class Stage3Refiner:
    """Stage 3: 单段 AD 精炼（无 recent-AD 上下文窗口）"""

    def __init__(
        self,
        llm_url: str = LLM_BASE_URL,
        llm_model: str = LLM_MODEL,
        llm_api_key: Optional[str] = None,
        before_window: int = 0,  # 保留参数兼容；默认 0=不使用
        canonical_characters: Optional[dict] = None,
    ):
        self.client = OpenAI(base_url=llm_url, api_key=llm_api_key or LLM_API_KEY)
        self.llm_model = llm_model
        self.before_window = 0  # 架构上固定关闭跨段 AD 记忆
        self.canonical_characters = canonical_characters or {}

    def _clean(self, text: str, max_words: Optional[int] = None) -> str:
        text = text.strip()
        for prefix in ("AD:", "Output:", "Description:", "Narration:", '"'):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].strip()
        if text.endswith('"'):
            text = text[:-1]
        text = re.sub(r"[\u4e00-\u9fff]+", "", text).strip()
        if max_words is not None and max_words > 0:
            words = text.split()
            if len(words) > max_words:
                text = " ".join(words[:max_words]).rstrip(",;:") + "."
        return text

    def _character_line(self, chars: str) -> str:
        raw = (chars or "").strip()
        if not raw:
            return "UNKNOWN (only use names that already appear in the visual text; do not invent)"
        if _characters_are_none(raw):
            return "NONE (do not invent or name any person)"
        return raw

    def refine_segment(
        self,
        seg: ADSegment,
        context: Optional[EnhancedGlobalContext] = None,
        recent_ads: Optional[List[str]] = None,  # 兼容旧调用签名，忽略
    ) -> str:
        del context, recent_ads  # 单段精炼不使用
        visual = ""
        if seg.visual_description:
            visual = seg.visual_description
        elif hasattr(seg, "to_text"):
            visual = seg.to_text()
        if not visual:
            parts = [p for p in [seg.setting, seg.characters, seg.visual_description] if p]
            visual = " ".join(parts)
        if not visual.strip():
            return ""

        gap_sec = max(0.0, float(seg.end_sec) - float(seg.start_sec))
        max_words = max_words_for_gap(gap_sec)
        resolved_chars = _resolve_characters(seg)
        no_names = _should_forbid_names(seg)
        user = STAGE3_USER.format(
            visual_description=visual.strip(),
            character_info=self._character_line(resolved_chars),
            gap_sec=gap_sec,
            max_words=max_words,
        )
        try:
            r = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": STAGE3_SYSTEM},
                    {"role": "user", "content": user},
                ],
                max_tokens=128,
                temperature=0.2,
            )
            cleaned = self._clean(r.choices[0].message.content or "", max_words=max_words)
            if len(cleaned) < 8:
                return _fallback_from_visual(visual, max_words, no_names=no_names)
            if no_names:
                if re.search(
                    r"\b(Tony|Stark|Rhodey|Obadiah|Christine|Everhart|Everheart|Hogan|Jimmy|Raza|Stane)\b",
                    cleaned,
                    flags=re.I,
                ):
                    return _fallback_from_visual(visual, max_words, no_names=True)
            return cleaned
        except Exception:
            return _fallback_from_visual(visual, max_words, no_names=no_names)

    def process(
        self,
        ad_segments: List[ADSegment],
        context: Optional[EnhancedGlobalContext] = None,
        reporter: Optional[ProgressReporter] = None,
        show_progress: bool = False,
    ) -> List[str]:
        ads: List[str] = []
        for seg in progress_iter(
            ad_segments,
            show=show_progress and reporter is None,
            reporter=reporter,
            stage="Stage3 AD",
            desc="Stage3 AD",
            unit="ad",
            total=len(ad_segments),
        ):
            ads.append(self.refine_segment(seg, context, None))
        return ads
