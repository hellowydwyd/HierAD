"""阿里云百炼 ASR：本地 wav → {start,end,text}。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from hierad.config import ASR_MODEL, get_dashscope_api_key


def _require_dashscope():
    try:
        import dashscope  # noqa: F401
        from dashscope.audio.asr import Recognition, RecognitionCallback
    except ImportError as e:
        raise ImportError(
            'DashScope ASR 需要: pip install "hierad[cloud]" 或 pip install dashscope'
        ) from e
    return dashscope, Recognition, RecognitionCallback


def _sentence_to_seg(sentence: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(sentence, dict):
        return None
    text = (sentence.get("text") or "").strip()
    if not text:
        return None
    start_ms = sentence.get("begin_time", sentence.get("beginTime", 0)) or 0
    end_ms = sentence.get("end_time", sentence.get("endTime", start_ms)) or start_ms
    return {
        "start": float(start_ms) / 1000.0,
        "end": float(end_ms) / 1000.0,
        "text": text,
    }


def transcribe_with_dashscope(
    audio_path: str,
    *,
    model: Optional[str] = None,
    sample_rate: int = 16000,
) -> List[Dict[str, Any]]:
    """
    用百炼实时识别读本地 wav，返回 HierAD 段列表。
    不依赖公网 file_urls。
    """
    dashscope, Recognition, RecognitionCallback = _require_dashscope()
    key = get_dashscope_api_key()
    if not key:
        raise ValueError("DashScope ASR 需要环境变量 DASHSCOPE_API_KEY")
    dashscope.api_key = key

    path = Path(audio_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"音频不存在: {path}")

    segments: List[Dict[str, Any]] = []

    class _Collector(RecognitionCallback):
        def on_event(self, result):  # type: ignore[no-untyped-def]
            try:
                sentence = result.get_sentence()
            except Exception:
                return
            if not sentence:
                return
            ended = True
            if hasattr(result, "is_sentence_end"):
                try:
                    ended = bool(result.is_sentence_end(sentence))
                except Exception:
                    ended = True
            if not ended:
                return
            seg = _sentence_to_seg(sentence)
            if seg:
                segments.append(seg)

    rec = Recognition(
        model=model or ASR_MODEL,
        format="wav",
        sample_rate=sample_rate,
        callback=_Collector(),
    )
    rec.call(str(path))

    # 去重：实时回调可能重复推送同一句
    unique: List[Dict[str, Any]] = []
    seen = set()
    for seg in segments:
        key_t = (round(seg["start"], 3), round(seg["end"], 3), seg["text"])
        if key_t in seen:
            continue
        seen.add(key_t)
        unique.append(seg)
    unique.sort(key=lambda x: x["start"])
    return unique
