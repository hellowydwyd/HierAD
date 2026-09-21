"""共享 VLM HTTP 客户端"""

import time
from pathlib import Path
from typing import Optional

import requests


class VLMClient:
    def __init__(
        self,
        vlm_url: str = "http://localhost:8002",
        vlm_type: str = "qwen2.5-vl",
        vlm_endpoint: str = "/prompted_inference",
        timeout: int = 180,
        retries: int = 3,
    ):
        self.vlm_url = vlm_url.rstrip("/")
        self.vlm_type = vlm_type
        ep = (vlm_endpoint or "/prompted_inference").strip()
        self.vlm_endpoint = ep if ep.startswith("/") else f"/{ep}"
        self.timeout = timeout
        self.retries = retries

    def call(self, video_path: str, system_prompt: str, user_prompt: str) -> str:
        abs_path = str(Path(video_path).resolve())
        data = {
            "video_path": abs_path,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
        if self.vlm_type in ("video-xl2", "videollama3"):
            data = {"video_path": abs_path, "prompt": f"{system_prompt}\n\n{user_prompt}"}
            if self.vlm_type == "video-xl2":
                data["user_prompt"] = data["prompt"]

        url = f"{self.vlm_url}{self.vlm_endpoint}"
        last_err: Optional[Exception] = None
        for _ in range(self.retries):
            try:
                r = requests.post(url, json=data, timeout=self.timeout)
                if r.status_code != 200:
                    raise ValueError(f"VLM HTTP {r.status_code}")
                j = r.json()
                if j.get("code", 0) != 0:
                    raise ValueError(f"VLM error: {j.get('msg', j.get('message', 'unknown'))}")
                out = j.get("data") or j.get("result") or j.get("output") or j.get("text")
                if out is not None:
                    return str(out).strip()
                raise ValueError("VLM response missing content field")
            except (requests.RequestException, ValueError) as e:
                last_err = e
                time.sleep(2)
        raise RuntimeError(f"VLM 调用失败: {last_err}")
