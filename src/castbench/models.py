from __future__ import annotations

import os
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from queue import Queue
from threading import BoundedSemaphore
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .common import AgentError, InfrastructureError, digest, finite_array, load_json, write_json
from .contracts import Contract, Episode, PositiveInt


class ModelProfile(Contract):
    name: Literal["chronos", "timesfm", "sundial", "moirai"]
    engine: Literal["chronos_original", "timesfm_2p5", "sundial", "moirai_1"]
    python_executable: str
    checkpoint: str
    revision: str
    checkpoint_digest: str
    max_context: PositiveInt
    max_horizon: PositiveInt
    min_context: PositiveInt = 8
    seed: int = Field(default=0, strict=True, ge=0, lt=2**32)
    num_samples: PositiveInt = 20
    point_method: Literal["median", "mean", "native"] = "median"
    device: str = "cpu"
    dtype: Literal["float32", "bfloat16"] = "float32"
    missing_policy: Literal["reject", "linear_internal"] = "reject"
    dependency_versions: dict[str, str]
    timeout_seconds: PositiveInt = 600
    moirai_patch_size: Literal[8, 16, 32, 64, 128] | None = None

    @model_validator(mode="after")
    def supported_profile(self):
        engines = {
            "chronos": "chronos_original",
            "timesfm": "timesfm_2p5",
            "sundial": "sundial",
            "moirai": "moirai_1",
        }
        if engines[self.name] != self.engine or self.min_context > self.max_context:
            raise ValueError("Inconsistent model profile")
        if self.name in {"timesfm", "moirai"} and self.dtype != "float32":
            raise ValueError("These worker adapters currently support float32 only")
        if self.name == "timesfm" and self.point_method != "native":
            raise ValueError("TimesFM worker returns the native point head; specify point_method=native")
        if self.name != "timesfm" and self.point_method == "native":
            raise ValueError("Sample-based workers require mean or median extraction")
        if self.device not in {"cpu", "cuda:0", "npu:0"}:
            raise ValueError(
                "Isolate one accelerator in the worker environment and address it as cuda:0 or npu:0"
            )
        if self.name == "moirai" and self.moirai_patch_size is None:
            self.moirai_patch_size = 16
        if self.name != "moirai" and self.moirai_patch_size is not None:
            raise ValueError("moirai_patch_size applies only to MOIRAI")
        return self


class ModelService:
    def __init__(
        self,
        profiles: list[ModelProfile],
        store,
        max_parallel_workers=None,
        npu_devices=None,
    ):
        self.profiles = {p.name: ModelProfile.model_validate(p.model_dump()) for p in profiles}
        if len(self.profiles) != len(profiles):
            raise ValueError("Duplicate model profiles")
        self.store = store
        if max_parallel_workers is not None and (
            isinstance(max_parallel_workers, bool)
            or not isinstance(max_parallel_workers, int)
            or max_parallel_workers < 1
        ):
            raise ValueError("max_parallel_workers must be a positive integer")
        self._worker_slots = (
            BoundedSemaphore(max_parallel_workers) if max_parallel_workers is not None else None
        )
        npu_devices = [] if npu_devices is None else list(npu_devices)
        if any(isinstance(device, bool) or not isinstance(device, int) or device < 0 for device in npu_devices):
            raise ValueError("npu_devices must contain nonnegative integer device IDs")
        if len(set(npu_devices)) != len(npu_devices):
            raise ValueError("npu_devices must not contain duplicates")
        self.npu_devices = tuple(npu_devices)
        self._npu_slots = Queue(maxsize=len(npu_devices)) if npu_devices else None
        for device in npu_devices:
            self._npu_slots.put(device)

    def preflight(self):
        if set(self.profiles) != {"chronos", "timesfm", "sundial", "moirai"}:
            raise InfrastructureError("Core live evaluation requires all four model profiles")
        for p in self.profiles.values():
            if not Path(p.python_executable).is_file():
                raise InfrastructureError(f"Missing interpreter: {p.name}")
            if not Path(p.checkpoint).is_dir() or not p.checkpoint_digest.startswith("sha256:"):
                raise InfrastructureError(f"Use a local, frozen checkpoint and its digest for {p.name}")
            if p.revision in {"main", "latest", ""} or not p.dependency_versions:
                raise InfrastructureError("Freeze weight revision and dependency versions before evaluation")
        if any(p.device == "npu:0" for p in self.profiles.values()) and not self.npu_devices:
            raise InfrastructureError("NPU model profiles require an explicit nonempty NPU device pool")

    def predict(self, name: str, episode: Episode):
        return self._predict(name, episode)

    def _predict(self, name: str, episode: Episode):
        if name not in self.profiles:
            raise InfrastructureError(f"Model profile not configured: {name}")
        p = self.profiles[name]
        h = episode.history.target_values
        total = episode.task.forecast_gap + episode.task.horizon_length
        if len(h) < p.min_context or total > p.max_horizon:
            return {
                "status": "not_applicable",
                "structured_result": {},
                "warnings": ["Outside frozen model support"],
            }
        values = np.array([np.nan if v is None else v for v in h[-p.max_context :]], dtype=float)
        valid = np.isfinite(values)
        if not valid.all():
            if p.missing_policy != "linear_internal" or valid.sum() < 2 or not valid[0] or not valid[-1]:
                return {
                    "status": "not_applicable",
                    "structured_result": {},
                    "warnings": ["Missing targets have no supported frozen repair"],
                }
            values = np.interp(np.arange(len(values)), np.flatnonzero(valid), values[valid])
        # This target-only request is the complete payload seen by the external worker.
        request = {
            "profile": p.model_dump(),
            "target_values": values.tolist(),
            "frequency": episode.task.frequency,
            "start_timestamp": episode.history.timestamps[-len(values)],
            "prediction_length": total,
        }
        key = digest(
            {
                "request": request,
                "timeline": episode.forecast_timeline.model_dump(),
                "worker_digest": digest(
                    Path(__file__).with_name("model_worker.py").read_text(encoding="utf-8")
                ),
            }
        )
        result = self.store.cached_model(key)
        if result is None:
            leased_npu = None
            worker_slot = self._worker_slots if self._worker_slots is not None else nullcontext()
            # Cache lookup deliberately precedes this gate: cached forecasts do not consume
            # scarce accelerator/worker capacity during an Agent evaluation.
            with worker_slot:
                try:
                    if p.device == "npu:0":
                        if self._npu_slots is None:
                            raise InfrastructureError(
                                "NPU worker requested without a configured device pool"
                            )
                        leased_npu = self._npu_slots.get()
                    with tempfile.TemporaryDirectory(prefix="castbench-model-") as temp:
                        inp, out = Path(temp) / "request.json", Path(temp) / "result.json"
                        write_json(inp, request)
                        env = {
                            k: v
                            for k, v in os.environ.items()
                            if not any(
                                s in k.upper()
                                for s in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")
                            )
                            and not k.upper().endswith("_KEY")
                        }
                        env.update(
                            HF_HUB_OFFLINE="1",
                            TRANSFORMERS_OFFLINE="1",
                            PYTHONIOENCODING="utf-8",
                            PYTHONUTF8="1",
                        )
                        if p.device == "cpu":
                            env["CUDA_VISIBLE_DEVICES"] = ""
                            env["ASCEND_RT_VISIBLE_DEVICES"] = ""
                        elif p.device == "cuda:0":
                            env["ASCEND_RT_VISIBLE_DEVICES"] = ""
                            # Preserve the existing server worktree's deterministic CUDA setup.
                            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                        else:
                            env["CUDA_VISIBLE_DEVICES"] = ""
                            env["ASCEND_RT_VISIBLE_DEVICES"] = str(leased_npu)
                        try:
                            proc = subprocess.run(
                                [
                                    p.python_executable,
                                    str(Path(__file__).with_name("model_worker.py")),
                                    str(inp),
                                    str(out),
                                ],
                                cwd=temp,
                                env=env,
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                timeout=p.timeout_seconds,
                                check=False,
                            )
                        except (OSError, subprocess.TimeoutExpired) as exc:
                            raise InfrastructureError(
                                f"{name} worker unavailable: {type(exc).__name__}"
                            ) from exc
                        if proc.returncode or not out.exists():
                            raise InfrastructureError(f"{name} worker failed: {proc.stderr[-4000:]}")
                        try:
                            result = load_json(out)
                            finite_array(result["point_values"], total)
                        except (AgentError, KeyError, ValueError, TypeError) as exc:
                            raise InfrastructureError(
                                "Model worker returned invalid output"
                            ) from exc
                        result = self.store.cached_model(key, result)
                finally:
                    if leased_npu is not None:
                        self._npu_slots.put(leased_npu)
        try:
            point = finite_array(result["point_values"], total)
        except (AgentError, KeyError, TypeError) as exc:
            raise InfrastructureError("Cached model result failed validation") from exc
        gap = episode.task.forecast_gap
        return {
            "status": "ok",
            "structured_result": {
                "gap_values": point[:gap].tolist(),
                "scored_values": point[gap:].tolist(),
                "model_card": p.model_dump(exclude={"python_executable", "checkpoint"}),
                "adaptation": {
                    "original_length": len(h),
                    "actual_length": len(values),
                    "dropped_points": max(0, len(h) - len(values)),
                    "imputed_indices": np.flatnonzero(~valid).tolist(),
                    "frequency": episode.task.frequency,
                    "seed": p.seed,
                    "point_method": p.point_method,
                },
                "distribution": result.get("distribution"),
                "cache_key": key,
            },
            "warnings": [],
        }
