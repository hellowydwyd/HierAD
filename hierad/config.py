"""
HierAD 统一配置

优先级: 环境变量 > config.yaml > 默认值
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ==================== 默认配置 ====================
_DEFAULTS = {
    "LLM_API_KEY": "EMPTY",
    "LLM_BASE_URL": "http://127.0.0.1:8014/v1",
    "LLM_MODEL": "llama3-8b",
    "VLM_URL": "http://localhost:8002",
    "VLM_TYPE": "qwen2.5-vl",
    "VLM_ENDPOINT": "/prompted_inference",
    "ASR_URL": "http://127.0.0.1:8001/asr_recognize",
    "ASR_DIARIZE": "false",
    "TTS_URL": "http://127.0.0.1:8003",
    "TTS_SPK_ID": "",
    "TTS_SPEED": "1.2",
    "TTS_VOICE_NAME": "康辉",
    "TTS_VOICE_ID": "",
    "VOICE_DB_PATH": "",
    "TRANSLATE_ZH": "true",
    "TMDB_API_KEY": "",
    "TMDB_PROXY": "",
    "ACTOR_DATABASES_DIR": str(PROJECT_ROOT / "actor_databases"),
}


def _load_yaml_config() -> Dict[str, Any]:
    """可选加载 config.yaml（项目根或当前目录）"""
    for base in (PROJECT_ROOT, Path.cwd()):
        path = base / "config.yaml"
        if path.exists():
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}


_YAML = _load_yaml_config()


# ==================== API 配置 ====================
LLM_API_KEY = os.getenv("HIERAD_LLM_API_KEY") or (_YAML.get("llm", {}) or {}).get("api_key") or _DEFAULTS["LLM_API_KEY"]
LLM_BASE_URL = os.getenv("HIERAD_LLM_BASE_URL") or (_YAML.get("llm", {}) or {}).get("base_url") or _DEFAULTS["LLM_BASE_URL"]
LLM_MODEL = os.getenv("HIERAD_LLM_MODEL") or (_YAML.get("llm", {}) or {}).get("model") or _DEFAULTS["LLM_MODEL"]

VLM_URL = os.getenv("HIERAD_VLM_URL") or (_YAML.get("vlm", {}) or {}).get("url") or _DEFAULTS["VLM_URL"]
VLM_TYPE = os.getenv("HIERAD_VLM_TYPE") or (_YAML.get("vlm", {}) or {}).get("type") or _DEFAULTS["VLM_TYPE"]
VLM_ENDPOINT = os.getenv("HIERAD_VLM_ENDPOINT") or (_YAML.get("vlm", {}) or {}).get("endpoint") or _DEFAULTS["VLM_ENDPOINT"]

# ASR（Whisper HTTP 服务）
ASR_URL = os.getenv("HIERAD_ASR_URL") or (_YAML.get("asr", {}) or {}).get("url") or _DEFAULTS["ASR_URL"]
_asr_diarize = os.getenv("HIERAD_ASR_DIARIZE") or (_YAML.get("asr", {}) or {}).get("diarize", _DEFAULTS["ASR_DIARIZE"])
ASR_DIARIZE = str(_asr_diarize).lower() in ("1", "true", "yes", "on")

# TTS（CosyVoice）
_tts_yaml = _YAML.get("tts", {}) or {}
TTS_URL = os.getenv("HIERAD_TTS_URL") or _tts_yaml.get("url") or _DEFAULTS["TTS_URL"]
TTS_SPK_ID = os.getenv("HIERAD_TTS_SPK_ID") or _tts_yaml.get("spk_id") or _DEFAULTS["TTS_SPK_ID"]
TTS_VOICE_NAME = (
    os.getenv("HIERAD_TTS_VOICE_NAME")
    or _tts_yaml.get("voice_name")
    or _DEFAULTS["TTS_VOICE_NAME"]
)
TTS_VOICE_ID = (
    os.getenv("HIERAD_TTS_VOICE_ID")
    or _tts_yaml.get("voice_id")
    or _DEFAULTS["TTS_VOICE_ID"]
)
VOICE_DB_PATH = (
    os.getenv("HIERAD_VOICE_DB")
    or _tts_yaml.get("voice_db")
    or (_YAML.get("paths", {}) or {}).get("voice_db")
    or _DEFAULTS["VOICE_DB_PATH"]
)
VOICE_NAME = TTS_VOICE_NAME  # alias for voice.resolve
try:
    TTS_SPEED = float(os.getenv("HIERAD_TTS_SPEED") or _tts_yaml.get("speed") or _DEFAULTS["TTS_SPEED"])
except (TypeError, ValueError):
    TTS_SPEED = float(_DEFAULTS["TTS_SPEED"])

_translate = (
    os.getenv("HIERAD_TRANSLATE_ZH")
    or (_YAML.get("export", {}) or {}).get("translate_zh")
    or _DEFAULTS["TRANSLATE_ZH"]
)
TRANSLATE_ZH = str(_translate).lower() in ("1", "true", "yes", "on")

# TMDB（预处理）
TMDB_API_KEY = os.getenv("HIERAD_TMDB_API_KEY") or (_YAML.get("tmdb", {}) or {}).get("api_key") or _DEFAULTS["TMDB_API_KEY"]
TMDB_PROXY = (
    os.getenv("HIERAD_TMDB_PROXY")
    or (_YAML.get("tmdb", {}) or {}).get("proxy")
    or os.getenv("HTTPS_PROXY")
    or os.getenv("HTTP_PROXY")
    or _DEFAULTS["TMDB_PROXY"]
)

# ==================== 路径配置 ====================
DEFAULT_WORK_DIR = PROJECT_ROOT / "work_dir"
ACTOR_DATABASES_DIR = Path(
    os.getenv("HIERAD_ACTOR_DB_DIR")
    or (_YAML.get("paths", {}) or {}).get("actor_databases")
    or _DEFAULTS["ACTOR_DATABASES_DIR"]
)


def get_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """获取 OpenAI 兼容的 LLM 客户端"""
    from openai import OpenAI
    return OpenAI(
        api_key=api_key or LLM_API_KEY,
        base_url=base_url or LLM_BASE_URL,
    )
