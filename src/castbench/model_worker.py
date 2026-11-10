"""Standalone target-only worker. Run under each model's isolated Python environment.

Only explicit local pinned weights are loaded. It never reads an Episode bundle.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def period_frequency(frequency):
    from pandas.tseries.frequencies import to_offset
    from pandas.tseries.offsets import MonthBegin

    offset = to_offset(frequency)
    # Period represents a calendar month, rather than a start/end timestamp.
    # The Episode retains MS; only GluonTS' internal Period label uses M.
    return f"{offset.n}M" if isinstance(offset, MonthBegin) else frequency


@contextmanager
def deterministic_sundial_median(torch):
    original = torch.Tensor.median

    def median(tensor, *args, **kwargs):
        if tensor.is_cuda and torch.are_deterministic_algorithms_enabled():
            result = original(tensor.cpu(), *args, **kwargs)
            if isinstance(result, tuple):
                return tuple(value.to(tensor.device) for value in result)
            return result.to(tensor.device)
        return original(tensor, *args, **kwargs)

    # Native generation only uses median values; CPU reduction preserves the
    # lower-median definition and avoids CUDA's unsupported indices operation.
    torch.Tensor.median = median
    try:
        yield
    finally:
        torch.Tensor.median = original


def checkpoint_hash(root):
    h = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            h.update(path.relative_to(root).as_posix().encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
    return "sha256:" + h.hexdigest()


def run(request):
    import numpy as np
    import torch

    p = request["profile"]
    for package, version in p["dependency_versions"].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f"Dependency mismatch: {package}")
    if checkpoint_hash(p["checkpoint"]) != p["checkpoint_digest"]:
        raise RuntimeError("Checkpoint content does not match frozen digest")
    expected = {
        "chronos": "chronos_original",
        "timesfm": "timesfm_2p5",
        "sundial": "sundial",
        "moirai": "moirai_1",
    }
    if expected[p["name"]] != p["engine"]:
        raise RuntimeError("Model name/engine mismatch")
    if p["device"] == "npu:0":
        # Register the Ascend backend. The parent exposes one physical device, so
        # every isolated worker consistently addresses it as logical npu:0.
        import torch_npu  # noqa: F401

    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    dtype = getattr(torch, p["dtype"])
    values = np.asarray(request["target_values"], dtype=np.float32)
    horizon = request["prediction_length"]
    distribution = None
    with torch.inference_mode():
        if p["engine"] == "chronos_original":
            from chronos import ChronosPipeline

            model = ChronosPipeline.from_pretrained(
                p["checkpoint"], device_map=p["device"], torch_dtype=dtype, local_files_only=True
            )
            samples = (
                model.predict(
                    torch.tensor(values),
                    prediction_length=horizon,
                    num_samples=p["num_samples"],
                    limit_prediction_length=False,
                )[0]
                .float()
                .cpu()
                .numpy()
            )
        elif p["engine"] == "timesfm_2p5":
            import timesfm

            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                p["checkpoint"], local_files_only=True, torch_compile=False
            )
            requested_device = torch.device(p["device"])
            if model.model.device != requested_device:
                # TimesFM 2.5 currently auto-selects CUDA or CPU in its constructor
                # and exposes no device argument. Override the fully loaded native
                # module before compile/decode so the benchmark's frozen profile is
                # authoritative on CUDA and Ascend alike.
                model.model.to(requested_device)
                model.model.device = requested_device
                model.model.device_count = 1
            model.compile(
                timesfm.ForecastConfig(
                    max_context=p["max_context"],
                    max_horizon=p["max_horizon"],
                    normalize_inputs=True,
                    per_core_batch_size=1,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=False,
                    fix_quantile_crossing=True,
                )
            )
            point, quantiles = model.forecast(horizon=horizon, inputs=[values])
            point = np.asarray(point[0], dtype=float)
            distribution = {
                "kind": "native_quantiles",
                "columns": ["mean", "q0.1", "q0.2", "q0.3", "q0.4", "q0.5", "q0.6", "q0.7", "q0.8", "q0.9"],
                "values": np.asarray(quantiles[0]).tolist(),
            }
            samples = None
        elif p["engine"] == "sundial":
            from transformers import AutoModelForCausalLM

            # Local repository code is executed only after its complete hash was checked.
            model = (
                AutoModelForCausalLM.from_pretrained(
                    p["checkpoint"], trust_remote_code=True, local_files_only=True
                )
                .to(device=p["device"], dtype=dtype)
                .eval()
            )
            with deterministic_sundial_median(torch):
                output = model.generate(
                    torch.tensor(values).unsqueeze(0).to(device=p["device"], dtype=dtype),
                    max_new_tokens=horizon,
                    num_samples=p["num_samples"],
                )
            samples = output[0].float().cpu().numpy()
        else:
            import pandas as pd
            from gluonts.dataset.common import ListDataset
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

            native_frequency = period_frequency(request["frequency"])
            model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(p["checkpoint"]),
                prediction_length=horizon,
                context_length=len(values),
                # Auto selection needs context+prediction history; these tasks provide only context.
                patch_size=p["moirai_patch_size"],
                num_samples=p["num_samples"],
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            dataset = ListDataset(
                [
                    {
                        "target": values,
                        "start": pd.Period(
                            pd.Timestamp(request["start_timestamp"]).tz_convert("UTC").tz_localize(None),
                            freq=native_frequency,
                        ),
                    }
                ],
                freq=native_frequency,
            )
            prediction = next(iter(model.create_predictor(batch_size=1, device=p["device"]).predict(dataset)))
            samples = np.asarray(prediction.samples)
        if samples is not None:
            if samples.ndim != 2 or samples.shape[1] != horizon:
                raise RuntimeError(f"Unexpected native sample shape: {samples.shape}")
            point = np.median(samples, axis=0) if p["point_method"] == "median" else np.mean(samples, axis=0)
            distribution = {"kind": "samples", "values": samples.tolist()}
    if point.shape != (horizon,) or not np.isfinite(point).all():
        raise RuntimeError("Native prediction is not a complete finite horizon")
    return {"point_values": point.tolist(), "distribution": distribution}


if __name__ == "__main__":
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = run(request)
    Path(sys.argv[2]).write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
