from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import EMOTIONS


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    text: str = Field(min_length=1, max_length=20000)
    speaker: str = Field(pattern=r"^[0-9a-f]{64}$", description="已上传音频的 SHA-256 asset ID")
    lang: Literal["ZH", "EN", "JA", "ES", "AR", "ZHEN"] = "ZH"
    emotion_audio: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    emotion_vector: list[float] | None = Field(default=None, min_length=8, max_length=8)
    emotion_text: str | None = Field(default=None, min_length=1, max_length=1000)
    emotion_auto: bool = False
    emotion_alpha: float = Field(default=1.0, ge=0, le=1)
    emotion_random: bool = False
    duration_factor: float = Field(default=1.0, ge=0.5, le=2.0)
    seed: int = Field(default=42, ge=0, le=4294967295)
    max_text_tokens_per_segment: int = Field(default=120, ge=20, le=200)
    interval_silence: int = Field(default=200, ge=0, le=3000)
    text_normalization: bool = True
    num_beams: int = Field(default=3, ge=1, le=5, description="默认3遵循上游；可测试1以减少beam搜索开销")
    temperature: float = Field(default=0.8, ge=0.1, le=2)
    top_p: float = Field(default=0.8, gt=0, le=1)
    top_k: int = Field(default=30, ge=1, le=100)
    max_mel_tokens: int = Field(default=1500, ge=50, le=3000)

    @model_validator(mode="after")
    def valid_controls(self):
        if not self.text.strip():
            raise ValueError("text 不能仅含空白")
        modes = [self.emotion_audio is not None, self.emotion_vector is not None,
                 self.emotion_text is not None, self.emotion_auto]
        if sum(modes) > 1:
            raise ValueError("情绪参考音频、向量、文字描述、自动情绪只能选择一种")
        if self.emotion_vector is not None:
            if any(v < 0 or v > 1 for v in self.emotion_vector):
                raise ValueError("八维情绪强度必须在 0..1")
            if sum(self.emotion_vector) > 1.000001:
                raise ValueError("情绪向量之和不能超过 1；建议总强度 0.6..0.8")
        return self


def capabilities(qwen=False, backend="indextts"):
    return {"model": "IndexTeam/IndexTTS-2.5", "backend": backend,
            "languages": ["ZH", "EN", "JA", "ES", "AR"], "mixed_text_mode": "ZHEN",
            "emotion_order": EMOTIONS, "text_emotion_enabled": qwen,
            "emotion_modes": ["speaker", "audio", "vector", "text", "auto"],
            "sample_rate": 22050, "format": "PCM WAV", "concurrency": 1,
            "duration_factor": {"min": 0.5, "max": 2.0, "greater_than_one": "slower"},
            "pronunciation": "<文字|PINYIN> / <word|CMU PHONEMES> / <文字|かな>",
            "progress": "upstream phase estimates; not a wall-clock ETA",
            "request_schema": JobRequest.model_json_schema()}
