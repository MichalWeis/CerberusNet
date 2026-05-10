"""
Benchmark CerberusNet (MultiModalFasterRCNN) inference time on 1000 samples
with batch size 1, for 1 / 2 / 3 input modalities.

Usage
-----
    python benchmark_inference.py \
        --cache-dir /path/to/preprocessed/cache \
        --checkpoint checkpoints/epoch_0050.pt \
        --num-samples 1000 \
        --warmup 50 \
        --scenarios rgb rgb+lidar rgb+lidar+radar aux

Add --synthetic if you have no cache dir; the script will fabricate tensors
matching the model's expected shapes.
"""

from __future__ import annotations

import argparse
import time
from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.detection.image_list import ImageList

from model import build_model
from dataset import PreprocessedSTFDataset, collate_fn_stf, STF_CLASSES


# ---------------------------------------------------------------------------
# Modality-aware forward
# ---------------------------------------------------------------------------

def get_compatible_heads(model, available: Sequence[str]) -> List[str]:
    """
    Return the head whose required modalities exactly match `available`.

    Exact match (not subset): with all 3 modalities present we run only the
    "all" head, not also the three pairwise heads. With 2 modalities we run
    the matching pairwise head. With 1 modality no head matches.
    """
    avail = set(available)
    return [
        name for name, req in model.fusion_combos.items()
        if set(req) == avail
    ]


@torch.no_grad()
def forward_selective(
    model,
    modalities_list: List[Dict[str, torch.Tensor]],
    available: Sequence[str],
):
    """
    Forward pass restricted to the given modalities.

    Runs:
      - stem + shared encoder only for `available` modalities,
      - detection heads whose required modalities are a subset of `available`.

    Mirrors MultiModalFasterRCNN.forward (eval branch) but with selective
    modality processing. Used for inference-time benchmarking.
    """
    device = next(model.parameters()).device
    batch_size = len(modalities_list)
    avail = list(available)
    if not avail:
        raise ValueError("`available` must contain at least one modality.")

    # --- original spatial size (use first available modality) ---
    ref = avail[0]
    orig_h, orig_w = modalities_list[0][ref].shape[-2:]
    new_h, new_w = model._target_size(orig_h, orig_w)

    # --- stack, normalize, resize only the requested modalities ---
    batches: Dict[str, torch.Tensor] = {}
    for name in avail:
        t = torch.stack([m[name] for m in modalities_list]).to(device, non_blocking=True)
        t = model._normalize(name, t)
        t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        batches[name] = t

    # --- stems + shared encoder for each requested modality ---
    features: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in avail:
        features[name] = model.encoder(model.stems[name](batches[name]))

    image_sizes = [(new_h, new_w)] * batch_size
    image_list = ImageList(batches[ref], image_sizes)

    # --- heads whose required modalities are all present ---
    heads_to_run = get_compatible_heads(model, avail)

    all_detections: Dict[str, list] = {}
    for head_name in heads_to_run:
        modality_keys = model.fusion_combos[head_name]
        fused = OrderedDict()
        for fpn_key in features[modality_keys[0]]:
            fused[fpn_key] = torch.cat(
                [features[m][fpn_key] for m in modality_keys], dim=1
            )
        detections, _ = model.heads[head_name](fused, image_list, None)
        all_detections[head_name] = detections

    return all_detections, heads_to_run


@torch.no_grad()
def forward_aux_only(model, modalities_list: List[Dict[str, torch.Tensor]]):
    """
    Forward pass restricted to the auxiliary weather/daytime head.

    Path: rgb -> rgb_stem -> encoder.layer1 -> encoder.layer2 -> weather_daytime_head.
    No layer3/layer4, no FPN, no RPN, no ROI, no other modalities.

    Inputs are resized to the same dims used by the detection branch so the
    measurement is comparable to the detection scenarios.
    """
    if not hasattr(model, "weather_daytime_head"):
        raise AttributeError(
            "Model has no `weather_daytime_head` — make sure you're using the "
            "model.py version that defines the auxiliary head."
        )

    device = next(model.parameters()).device
    batch_size = len(modalities_list)

    orig_h, orig_w = modalities_list[0]["rgb"].shape[-2:]
    new_h, new_w = model._target_size(orig_h, orig_w)

    rgb = torch.stack([m["rgb"] for m in modalities_list]).to(device, non_blocking=True)
    rgb = model._normalize("rgb", rgb)
    rgb = F.interpolate(rgb, size=(new_h, new_w), mode="bilinear", align_corners=False)

    stem_out = model.stems["rgb"](rgb)
    c1 = model.encoder.layer1(stem_out)
    c2 = model.encoder.layer2(c1)

    aux_out = model.weather_daytime_head(c2)
    return aux_out


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_one_call(thunk, use_cuda: bool) -> float:
    """Return elapsed wall time of `thunk()` in milliseconds.

    `thunk` is a zero-arg callable that performs the forward pass. Using a
    thunk lets the same timer time either the detection or aux path without
    knowing which one it is.
    """
    if use_cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        thunk()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        thunk()
        return (time.perf_counter() - t0) * 1000.0


def benchmark(
    model,
    sample_provider,
    make_thunk,
    num_samples: int,
    warmup: int,
    device: str,
) -> List[float]:
    """
    Run `warmup` warm-up iterations then time `num_samples` forward passes.

    `sample_provider()` returns one batch (list of modality dicts of size 1).
    `make_thunk(modalities)` returns a zero-arg callable that runs the forward
    pass on those modalities — caller decides whether it's detection or aux.
    """
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    model.eval()

    # warmup (not measured)
    for _ in range(warmup):
        modalities = sample_provider()
        with torch.no_grad():
            make_thunk(modalities)()
    if use_cuda:
        torch.cuda.synchronize()

    times_ms: List[float] = []
    for _ in range(num_samples):
        modalities = sample_provider()
        with torch.no_grad():
            t = time_one_call(make_thunk(modalities), use_cuda)
        times_ms.append(t)
    return times_ms


# ---------------------------------------------------------------------------
# Sample providers
# ---------------------------------------------------------------------------

def make_loader_provider(loader):
    """Yield successive batches; cycle when exhausted."""
    iterator = iter(loader)

    def _next():
        nonlocal iterator
        try:
            modalities, _targets, _conds = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            modalities, _targets, _conds = next(iterator)
        return list(modalities)

    return _next


def make_synthetic_provider(h: int = 1024, w: int = 1920):
    """Return random tensors of plausible shapes (no disk I/O)."""
    def _next():
        return [{
            "rgb":   torch.rand(3, h, w),
            "lidar": torch.rand(2, h, w),
            "radar": torch.rand(2, h, w),
        }]
    return _next


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(name: str, heads: List[str], times_ms: List[float]) -> dict:
    arr = np.asarray(times_ms)
    summary = {
        "scenario": name,
        "heads_run": heads,
        "n":      int(arr.size),
        "mean":   float(arr.mean()),
        "median": float(np.median(arr)),
        "std":    float(arr.std()),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
        "fps":    float(1000.0 / arr.mean()),
    }
    print(f"\n=== {name} ===")
    print(f"  heads run : {heads if heads else '(none — stem+encoder only)'}")
    print(f"  samples   : {summary['n']}")
    print(f"  mean      : {summary['mean']:.2f} ms")
    print(f"  median    : {summary['median']:.2f} ms")
    print(f"  std       : {summary['std']:.2f} ms")
    print(f"  min / max : {summary['min']:.2f} / {summary['max']:.2f} ms")
    print(f"  p95 / p99 : {summary['p95']:.2f} / {summary['p99']:.2f} ms")
    print(f"  throughput: {summary['fps']:.2f} fps")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Directory of preprocessed .pt samples.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional model checkpoint (.pt).")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--synthetic", action="store_true",
                   help="Use random synthetic inputs instead of dataset.")
    p.add_argument("--syn-h", type=int, default=1024)
    p.add_argument("--syn-w", type=int, default=1920)
    p.add_argument(
        "--scenarios", type=str, nargs="+",
        default=["rgb", "rgb+lidar", "rgb+lidar+radar", "aux"],
        help="Plus-separated modality lists, e.g. 'rgb+lidar lidar+radar'. "
             "Use the special keyword 'aux' to time the auxiliary "
             "weather/daytime head path (rgb stem + layer1 + layer2 + aux head).",
    )
    args = p.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- model ---
    model = build_model(
        num_classes=len(STF_CLASSES) + 1,
        eval_head="all",
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided — using freshly-initialized weights "
              "(timing is unaffected by weight values).")

    model.eval()

    # --- sample provider ---
    if args.synthetic or args.cache_dir is None:
        if not args.synthetic:
            print("[warn] no --cache-dir, falling back to synthetic input.")
        provider = make_synthetic_provider(args.syn_h, args.syn_w)
        print(f"Using synthetic input: 1x(rgb 3ch, lidar 2ch, radar 2ch) "
              f"@ {args.syn_h}x{args.syn_w}")
    else:
        ds = PreprocessedSTFDataset(cache_dir=args.cache_dir)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            collate_fn=collate_fn_stf, num_workers=0)
        provider = make_loader_provider(loader)
        print(f"Loaded {len(ds)} samples from {args.cache_dir}")

    # --- run scenarios ---
    summaries = []
    for scen in args.scenarios:
        # --- aux head: special scenario, separate forward path ---
        if scen.strip().lower() == "aux":
            if not hasattr(model, "weather_daytime_head"):
                print(f"\n[skip] scenario 'aux' requested but model has no "
                      f"`weather_daytime_head`. Update model.py or remove 'aux'.")
                continue
            label = "aux head: rgb (stem+layer1+layer2)"
            heads_run = ["weather_daytime_head"]
            print(f"\n>>> {label}")
            print(f"    path: rgb_stem -> layer1 -> layer2 -> weather_daytime_head")

            def make_thunk(mods):
                return lambda: forward_aux_only(model, mods)

            times = benchmark(
                model, provider, make_thunk,
                num_samples=args.num_samples,
                warmup=args.warmup,
                device=device,
            )
            summaries.append(summarize(label, heads_run, times))
            continue

        # --- detection scenarios ---
        mods = [m.strip() for m in scen.split("+") if m.strip()]
        unknown = [m for m in mods if m not in {"rgb", "lidar", "radar"}]
        if unknown:
            print(f"\n[skip] unknown modalities in '{scen}': {unknown}")
            continue

        compat = get_compatible_heads(model, mods)
        n = len(mods)
        suffix = "ies" if n != 1 else "y"
        label = f"{n} modalit{suffix}: {'+'.join(mods)}"
        print(f"\n>>> {label}")
        print(f"    compatible heads: {compat if compat else '(none)'}")

        # bind `mods` per-iteration via default arg to avoid late-binding bug
        def make_thunk(modalities, mods=mods):
            return lambda: forward_selective(model, modalities, mods)

        times = benchmark(
            model, provider, make_thunk,
            num_samples=args.num_samples,
            warmup=args.warmup,
            device=device,
        )
        summaries.append(summarize(label, compat, times))

    # --- final table ---
    print("\n\n" + "=" * 78)
    print(f"{'scenario':<32}{'heads':<22}{'mean (ms)':>12}{'median':>10}{'fps':>10}")
    print("-" * 78)
    for s in summaries:
        heads_s = ",".join(s["heads_run"]) if s["heads_run"] else "(none)"
        if len(heads_s) > 20:
            heads_s = heads_s[:17] + "..."
        print(f"{s['scenario']:<32}{heads_s:<22}{s['mean']:>12.2f}"
              f"{s['median']:>10.2f}{s['fps']:>10.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()