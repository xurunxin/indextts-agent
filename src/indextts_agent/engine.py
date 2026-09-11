import os
import random
import sys
import time
import warnings
from pathlib import Path

from .common import Failure


class Cancelled(Exception):
    pass


class IndexEngine:
    def __init__(self, config):
        upstream = Path(config["upstream"]).resolve()
        sys.path.insert(0, str(upstream))
        # Upstream has relative resource paths. This process owns exactly one engine.
        os.chdir(upstream)
        import torch
        if config["device"].startswith("cuda") and not torch.cuda.is_available():
            raise Failure("CUDA_UNAVAILABLE", "CUDA 不可用；运行 itt doctor --runtime-python 检查")
        from indextts.infer_v2_5 import IndexTTS2
        model_dir = str(Path(config["model_dir"]).resolve())
        from indextts.utils.model_download import ensure_models_available
        ensure_models_available(model_dir)
        self.model = IndexTTS2(cfg_path=str(Path(model_dir) / "config.yaml"), model_dir=model_dir,
                               device=config["device"], use_bf16=config["bf16"],
                               use_cuda_kernel=config["cuda_kernel"], use_deepspeed=False,
                               use_accel=False, use_torch_compile=config["compile"],
                               use_qwen_emo=config["qwen_emotion"])

    def generate(self, request, speaker, emotion, output, progress):
        import numpy as np
        import torch
        random.seed(request["seed"])
        np.random.seed(request["seed"])
        torch.manual_seed(request["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request["seed"])
            torch.cuda.reset_peak_memory_stats()
        self.model.gr_progress = progress
        kwargs = {key: request[key] for key in ("lang", "duration_factor", "max_text_tokens_per_segment",
                  "interval_silence", "text_normalization", "num_beams", "temperature", "top_p", "top_k", "max_mel_tokens")}
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", RuntimeWarning)
                self.model.infer(spk_audio_prompt=str(speaker), text=request["text"], output_path=str(output),
                                 emo_audio_prompt=str(emotion) if emotion else None,
                                 emo_vector=request["emotion_vector"], emo_alpha=request["emotion_alpha"],
                                 use_emo_text=request["emotion_auto"] or request["emotion_text"] is not None,
                                 emo_text=request["emotion_text"], use_random=request["emotion_random"], **kwargs)
            return {"warnings": list(dict.fromkeys(str(w.message) for w in captured if issubclass(w.category, RuntimeWarning))),
                    "peak_gpu_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if torch.cuda.is_available() else None}
        finally:
            self.model.gr_progress = None


class TestEngine:
    """Explicit transport fixture; never selected as a fallback for model failure."""
    def __init__(self, config):
        pass

    def generate(self, request, speaker, emotion, output, progress):
        import math
        import struct
        import wave
        for i in range(5):
            progress(i / 5, desc="test tone (not speech)")
            time.sleep(0.08)
        with wave.open(str(output), "wb") as f:
            f.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
            f.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(i * 440 * 2 * math.pi / 22050))) for i in range(2205)))
        return {"warnings": ["TEST BACKEND: synthetic tone, not IndexTTS inference"]}
