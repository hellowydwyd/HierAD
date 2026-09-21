"""
Stage 1: 全片场景理解

1. VLM × N：annotated scene clip + 前 m 窗分析 + clip ASR
2. LLM 增强：角色图谱、事件链、故事结构
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from hierad.config import LLM_BASE_URL, LLM_MODEL
from hierad.progress import ProgressReporter, progress_iter

from .models import ClipDescription, EnhancedGlobalContext
from .prompts import STAGE1_SCENE_SYSTEM, STAGE1_SCENE_USER_FIRST, STAGE1_SCENE_USER
from .vlm_client import VLMClient
from .utils import parse_stage1_output, normalize_character_with_mapping
from .stage1_enhance import Stage1Enhancer


def _parse_fallback(raw: str) -> Dict[str, str]:
    raw = raw.strip()
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            return {
                "setting": str(d.get("setting", "")).strip(),
                "characters": str(d.get("characters", "")).strip(),
                "action": str(d.get("action", "")).strip(),
                "raw": raw,
            }
        except json.JSONDecodeError:
            pass
    return {"setting": "", "characters": "", "action": raw[:200], "raw": raw}


class Stage1SceneUnderstanding:
    """Stage 1: 场景 clip VLM + LLM 全局增强"""

    def __init__(
        self,
        vlm_url: str = "http://localhost:8002",
        vlm_type: str = "qwen2.5-vl",
        vlm_endpoint: str = "/prompted_inference",
        llm_url: str = LLM_BASE_URL,
        llm_model: str = LLM_MODEL,
        llm_api_key: Optional[str] = None,
        context_window: int = 3,
        canonical_characters: Optional[Dict[str, str]] = None,
    ):
        self.vlm = VLMClient(vlm_url, vlm_type, vlm_endpoint)
        self.context_window = context_window
        self.canonical_characters = canonical_characters or {}
        self.enhancer = Stage1Enhancer(
            llm_url=llm_url,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            canonical_mapping=canonical_characters,
        )

    def _build_recent_context(self, clips: List[ClipDescription]) -> str:
        if not clips:
            return "(Opening — no previous scenes)"
        parts = []
        for c in clips[-self.context_window :]:
            if c.action:
                parts.append(f"- Clip {c.clip_idx}: {c.action[:100]}")
        return "\n".join(parts) if parts else "(No previous actions recorded)"

    def describe_clips(
        self,
        clip_manifest: List[dict],
        reporter: Optional[ProgressReporter] = None,
        show_progress: bool = False,
    ) -> List[ClipDescription]:
        """VLM 逐场景 clip 结构化描述"""
        descriptions: List[ClipDescription] = []
        for item in progress_iter(
            clip_manifest,
            show=show_progress and reporter is None,
            reporter=reporter,
            stage="Stage1 VLM",
            desc="Stage1 场景 VLM",
            unit="clip",
            total=len(clip_manifest),
        ):
            idx = int(item["clip_idx"])
            tc = item.get("timecode", "")
            tc_parts = tc.split(" --> ") if " --> " in tc else ["", ""]
            dialogue = item.get("dialogue_overlap", "(None)")
            recent = self._build_recent_context(descriptions)

            fmt = dict(recent_context=recent, dialogue_overlap=dialogue or "(None)")
            user = STAGE1_SCENE_USER_FIRST.format(**fmt) if idx == 0 else STAGE1_SCENE_USER.format(**fmt)

            try:
                raw = self.vlm.call(item["video_path"], STAGE1_SCENE_SYSTEM, user)
                parsed = parse_stage1_output(raw)
                if not parsed.get("action") and not parsed.get("setting"):
                    parsed = _parse_fallback(raw)
                chars = parsed.get("characters", "")
                if self.canonical_characters and chars:
                    parts = [p.strip() for p in re.split(r"[,;]", chars) if p.strip()]
                    chars = ", ".join(
                        normalize_character_with_mapping(p, self.canonical_characters)
                        for p in parts
                    )
                desc = ClipDescription(
                    clip_idx=idx,
                    timecode_start=tc_parts[0],
                    timecode_end=tc_parts[1] if len(tc_parts) > 1 else "",
                    start_sec=float(item.get("start_sec", 0)),
                    end_sec=float(item.get("end_sec", 0)),
                    setting=parsed.get("setting", ""),
                    characters=chars,
                    action=parsed.get("action", ""),
                    raw=parsed.get("raw", raw),
                    dialogue_overlap=dialogue,
                    video_path=item.get("video_path", ""),
                )
            except Exception:
                desc = ClipDescription(
                    clip_idx=idx,
                    timecode_start=tc_parts[0],
                    timecode_end=tc_parts[1] if len(tc_parts) > 1 else "",
                    start_sec=float(item.get("start_sec", 0)),
                    end_sec=float(item.get("end_sec", 0)),
                    setting="",
                    characters="",
                    action="",
                    raw="",
                    dialogue_overlap=dialogue,
                    video_path=item.get("video_path", ""),
                )
            descriptions.append(desc)
        return descriptions

    def process(
        self,
        clip_manifest: List[dict],
        reporter: Optional[ProgressReporter] = None,
        show_progress: bool = False,
    ) -> tuple[List[ClipDescription], EnhancedGlobalContext]:
        clips = self.describe_clips(
            clip_manifest, reporter=reporter, show_progress=show_progress
        )
        if reporter:
            reporter.stage("Stage1 增强", message="LLM 全局上下文")
        context = self.enhancer.enhance(clips, reporter=reporter, show_progress=show_progress)
        return clips, context


# 兼容旧名
Stage1Descriptor = Stage1SceneUnderstanding
