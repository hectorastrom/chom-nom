# @Time    : 2026-02-14
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : inference_harness.py

"""
Continuous inference harness for comparing FP32 .pt2 vs INT8 .pte YOLO models.

Loads a single model, feeds COCO128 images through it as fast as possible,
and reports real-time latency / throughput stats.  Run once per variant and
compare the final summaries side-by-side.

Supports:
  - .pt2  (torch.export) -- FP32 or INT8 q/dq graph
  - .pte  (ExecuTorch)   -- XNNPACK-lowered INT8, runs natively on M2

Usage:
    python -m compression.inference_harness weights/yolov8s.pt2
    python -m compression.inference_harness weights/yolov8s_int8_xnnpack.pte
    python -m compression.inference_harness weights/yolov8s.pt2 --warmup 20
"""

import argparse
import sys
import time
from pathlib import Path
from collections import deque

import torch
import torchvision.transforms as T
from PIL import Image

# Register quantized ops so INT8 .pt2 files can be loaded
from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: F401
    prepare_pt2e,
    convert_pt2e,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZE = 640
DATA_DIR = Path("data")
COCO128_IMAGES = DATA_DIR / "coco128" / "images" / "train2017"

W = 60  # column width for formatting


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_images(
    image_dir: Path,
    num_images: int,
    image_size: int,
) -> list[torch.Tensor]:
    """Load COCO128 images as (1, 3, H, W) float tensors in [0, 1]."""
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    paths = sorted(image_dir.glob("*.jpg"))[:num_images]
    if not paths:
        raise FileNotFoundError(
            f"No .jpg images found in {image_dir}.\n"
            "  Run: python -m compression.prepare_yolov8 --step download"
        )
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img).unsqueeze(0))
    return tensors


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_pt2_model(model_path: str):
    """Load a torch.export .pt2 program (FP32 or INT8 q/dq)."""
    ep = torch.export.load(model_path)
    model = ep.module()

    @torch.no_grad()
    def predict(x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    return predict, "pt2"


def load_pte_model(model_path: str):
    """Load an ExecuTorch .pte program via the ExecuTorch runtime."""
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(model_path)
    method = program.load_method("forward")

    def predict(x: torch.Tensor) -> torch.Tensor:
        # ExecuTorch expects a list of input tensors (contiguous)
        inputs = [x.contiguous()]
        outputs = method.execute(inputs)
        out = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return out

    return predict, "pte"


def load_model(model_path: str):
    """Auto-detect format and return (predict_fn, format_label)."""
    suffix = Path(model_path).suffix
    if suffix == ".pt2":
        return load_pt2_model(model_path)
    elif suffix == ".pte":
        return load_pte_model(model_path)
    else:
        raise ValueError(
            f"Unsupported model format '{suffix}'. Expected .pt2 or .pte"
        )


# ---------------------------------------------------------------------------
# Output summary helper
# ---------------------------------------------------------------------------

def summarize_output(result) -> str:
    """One-line summary of raw tensor output."""
    if isinstance(result, (tuple, list)):
        shapes = [
            str(tuple(t.shape))
            for t in result
            if isinstance(t, torch.Tensor)
        ]
        return f"outputs: {', '.join(shapes)}"
    if isinstance(result, torch.Tensor):
        return f"shape {tuple(result.shape)}"
    return str(type(result).__name__)


# ---------------------------------------------------------------------------
# Stats tracker
# ---------------------------------------------------------------------------

class FrameStats:
    """Rolling statistics for frame inference times."""

    def __init__(self, window_size: int = 100):
        self.times: deque[float] = deque(maxlen=window_size)
        self.all_times: list[float] = []
        self.total_frames = 0

    def record(self, elapsed_ms: float) -> None:
        self.times.append(elapsed_ms)
        self.all_times.append(elapsed_ms)
        self.total_frames += 1

    @property
    def rolling_fps(self) -> float:
        if not self.times:
            return 0.0
        avg_ms = sum(self.times) / len(self.times)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    @property
    def rolling_latency_ms(self) -> float:
        if not self.times:
            return 0.0
        return sum(self.times) / len(self.times)

    @property
    def global_fps(self) -> float:
        if not self.all_times:
            return 0.0
        avg_ms = sum(self.all_times) / len(self.all_times)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.all_times) if self.all_times else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.all_times) if self.all_times else 0.0

    @property
    def p50_ms(self) -> float:
        return self._percentile(50)

    @property
    def p95_ms(self) -> float:
        return self._percentile(95)

    @property
    def p99_ms(self) -> float:
        return self._percentile(99)

    def _percentile(self, p: float) -> float:
        if not self.all_times:
            return 0.0
        s = sorted(self.all_times)
        idx = int(len(s) * p / 100)
        idx = min(idx, len(s) - 1)
        return s[idx]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print()
    print("+" + "-" * (W - 2) + "+")
    print("|" + title.center(W - 2) + "|")
    print("+" + "-" * (W - 2) + "+")


def _kv(key: str, value: str, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{key:<24s}{value}")


def format_live_line(
    frame_idx: int,
    image_idx: int,
    latency_ms: float,
    stats: FrameStats,
    detail: str,
) -> str:
    """Single-line live status update (overwrites in-place)."""
    return (
        f"\r  frame {frame_idx:>6d} | "
        f"img {image_idx:>3d} | "
        f"{latency_ms:6.1f} ms | "
        f"avg {stats.rolling_latency_ms:6.1f} ms | "
        f"{stats.rolling_fps:6.1f} FPS | "
        f"{detail[:40]:<40s}"
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_harness(
    model_path: str,
    image_dir: str = str(COCO128_IMAGES),
    num_images: int = 128,
    warmup: int = 10,
) -> None:
    """Run continuous inference and report live timing stats.

    Loops over loaded images indefinitely (wrapping around) until
    interrupted with Ctrl+C, then prints a final summary.

    Args:
        model_path: Path to a .pt2 or .pte YOLO model.
        image_dir: Directory of .jpg images to feed through the model.
        num_images: Max images to load from disk.
        warmup: Number of warmup frames (excluded from stats).
    """
    # -- Setup ---------------------------------------------------------------
    _banner("YOLO Inference Harness")
    _kv("Model:", model_path)
    _kv("Format:", Path(model_path).suffix)
    _kv("Image dir:", image_dir)

    model_mb = Path(model_path).stat().st_size / (1024 * 1024)
    _kv("Model size:", f"{model_mb:.2f} MB")

    print()
    print("  Loading model...")
    predict_fn, fmt = load_model(model_path)
    print(f"  Model loaded ({fmt} backend)")

    print("  Loading images...")
    images = load_images(Path(image_dir), num_images, IMAGE_SIZE)
    print(f"  Loaded {len(images)} images ({IMAGE_SIZE}x{IMAGE_SIZE})")

    # -- Warmup --------------------------------------------------------------
    print(f"\n  Warmup ({warmup} frames)...")
    for i in range(warmup):
        img = images[i % len(images)]
        _ = predict_fn(img)
    print("  Warmup complete.\n")

    # -- Continuous inference -------------------------------------------------
    print("-" * W)
    print("  Running continuous inference (Ctrl+C to stop)")
    print("-" * W)

    stats = FrameStats(window_size=100)
    frame_idx = 0

    try:
        while True:
            img_idx = frame_idx % len(images)
            img = images[img_idx]

            t0 = time.perf_counter()
            result = predict_fn(img)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            stats.record(elapsed_ms)

            detail = summarize_output(result)
            line = format_live_line(
                frame_idx, img_idx, elapsed_ms, stats, detail,
            )
            print(line, end="", flush=True)

            frame_idx += 1

    except KeyboardInterrupt:
        pass

    # -- Final summary -------------------------------------------------------
    print()
    _banner("FINAL RESULTS")
    print()
    _kv("Model:", model_path)
    _kv("Format:", fmt)
    _kv("Model size:", f"{model_mb:.2f} MB")
    _kv("Total frames:", str(stats.total_frames))

    print()
    print("  Latency")
    _kv("Mean:", f"{stats.rolling_latency_ms:.2f} ms", indent=4)
    _kv("Min:", f"{stats.min_ms:.2f} ms", indent=4)
    _kv("Max:", f"{stats.max_ms:.2f} ms", indent=4)
    _kv("p50:", f"{stats.p50_ms:.2f} ms", indent=4)
    _kv("p95:", f"{stats.p95_ms:.2f} ms", indent=4)
    _kv("p99:", f"{stats.p99_ms:.2f} ms", indent=4)

    print()
    print("  Throughput")
    _kv("Rolling FPS:", f"{stats.rolling_fps:.1f}", indent=4)
    _kv("Overall FPS:", f"{stats.global_fps:.1f}", indent=4)

    print()
    print("+" + "-" * (W - 2) + "+")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Continuous YOLO inference harness -- "
            "compare .pt2 (FP32) vs .pte (INT8 XNNPACK) latency and output"
        ),
    )
    parser.add_argument(
        "model",
        help="Path to the YOLO model (.pt2 or .pte)",
    )
    parser.add_argument(
        "--image-dir",
        default=str(COCO128_IMAGES),
        help=f"Directory of .jpg images (default: {COCO128_IMAGES})",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=128,
        help="Max images to load (default: 128)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup frames before measurement (default: 10)",
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    run_harness(
        model_path=args.model,
        image_dir=args.image_dir,
        num_images=args.num_images,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
