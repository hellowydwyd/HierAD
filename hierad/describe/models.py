"""
描述模块数据结构
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


SETTING_SKIP_VALUES = frozenset([
    "same", "unchanged", "continues", "no change", "as before",
    "continuing", "still", "-", "n/a", "na", "",
])


@dataclass
class ClipDescription:
    """Stage1 场景 clip 结构化描述"""
    clip_idx: int
    timecode_start: str
    timecode_end: str
    start_sec: float
    end_sec: float
    setting: str
    characters: str
    action: str
    raw: str
    dialogue_overlap: str = ""
    video_path: str = ""

    def to_text(self) -> str:
        parts = []
        if self.setting and self.setting.lower().strip() not in SETTING_SKIP_VALUES:
            parts.append(f"[{self.setting}]")
        if self.characters and self.characters.lower().strip() not in ("none", "n/a", "-", ""):
            parts.append(self.characters)
        if self.action:
            parts.append(self.action)
        return " ".join(parts) if parts else self.raw

    def is_complete(self) -> bool:
        return bool((self.action or "").strip() or (self.setting or "").strip())

    def to_dict(self) -> dict:
        return {
            "clip_idx": self.clip_idx,
            "timecode": f"{self.timecode_start} --> {self.timecode_end}",
            "timecode_start": self.timecode_start,
            "timecode_end": self.timecode_end,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "setting": self.setting,
            "characters": self.characters,
            "action": self.action,
            "raw": self.raw,
            "dialogue_overlap": self.dialogue_overlap,
            "video_path": self.video_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClipDescription":
        tc = data.get("timecode") or ""
        parts = tc.split(" --> ") if " --> " in tc else ["", ""]
        return cls(
            clip_idx=int(data.get("clip_idx", 0)),
            timecode_start=str(data.get("timecode_start") or (parts[0] if parts else "")),
            timecode_end=str(data.get("timecode_end") or (parts[1] if len(parts) > 1 else "")),
            start_sec=float(data.get("start_sec", 0)),
            end_sec=float(data.get("end_sec", 0)),
            setting=str(data.get("setting") or ""),
            characters=str(data.get("characters") or ""),
            action=str(data.get("action") or ""),
            raw=str(data.get("raw") or ""),
            dialogue_overlap=str(data.get("dialogue_overlap") or ""),
            video_path=str(data.get("video_path") or ""),
        )


# 兼容旧名
DenseDescription = ClipDescription


@dataclass
class ADSegment:
    """Stage2 AD 片段分析结果"""
    ad_idx: int
    timecode_start: str
    timecode_end: str
    start_sec: float
    end_sec: float
    scene_idx: int
    scene_indices: List[int]
    setting: str
    characters: str
    visual_description: str
    source: str  # stage1_reuse | vlm
    dialogue_before: str = ""
    dialogue_after: str = ""
    video_path: str = ""
    raw: str = ""


@dataclass
class SceneInfo:
    """场景/clip 摘要（Stage1 LLM 增强）"""
    scene_id: int
    start_clip: int
    end_clip: int
    start_time: str
    end_time: str
    event: str = ""
    progression: str = ""
    progression_list: List[str] = field(default_factory=list)
    characters: str = ""

    # 兼容旧字段名
    @property
    def start_segment(self) -> int:
        return self.start_clip

    @property
    def end_segment(self) -> int:
        return self.end_clip


@dataclass
class StoryEvent:
    event_id: int
    clip_range: Tuple[int, int]
    description: str
    characters: List[str] = field(default_factory=list)
    causes: List[int] = field(default_factory=list)
    is_turning_point: bool = False
    importance: int = 1

    @property
    def segment_range(self) -> Tuple[int, int]:
        return self.clip_range


@dataclass
class StoryPhase:
    name: str
    event_ids: List[int] = field(default_factory=list)
    clip_range: Tuple[int, int] = (0, 0)
    summary: str = ""

    @property
    def segment_range(self) -> Tuple[int, int]:
        return self.clip_range


@dataclass
class StoryStructure:
    setting: str = ""
    core_conflict: str = ""
    phases: List[StoryPhase] = field(default_factory=list)
    turning_point_ids: List[int] = field(default_factory=list)


@dataclass
class Character:
    name: str
    description: str = ""
    first_appearance: int = 0
    role: str = "supporting"


@dataclass
class CharacterGraph:
    characters: dict = field(default_factory=dict)
    relations: List[dict] = field(default_factory=list)


@dataclass
class EnhancedGlobalContext:
    """Stage1 LLM 增强后的全局上下文"""
    scenes: List[SceneInfo] = field(default_factory=list)
    character_graph: CharacterGraph = field(default_factory=CharacterGraph)
    events: List[StoryEvent] = field(default_factory=list)
    story_structure: StoryStructure = field(default_factory=StoryStructure)
    clip_summaries: List[str] = field(default_factory=list)

    def get_scene_for_clip(self, clip_idx: int) -> Optional[SceneInfo]:
        for s in self.scenes:
            if s.start_clip <= clip_idx <= s.end_clip:
                return s
        return None

    def get_scene_for_segment(self, segment_idx: int) -> Optional[SceneInfo]:
        return self.get_scene_for_clip(segment_idx)
