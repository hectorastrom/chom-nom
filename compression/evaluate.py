# @Time    : 2026-02-14
# @Author  : Hector Astrom
# @File    : evaluate.py

"""
Evaluate accuracy and size differences between the original FP32 and
INT8-quantized YOLOv8s models.

Metrics:
  - Model size (state_dict on disk) before and after quantization
  - Per-image cosine similarity between FP32 and INT8 outputs
  - Mean squared error between outputs
  - Top-prediction agreement rate

Usage:
    python -m compression.evaluate
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# Ensure quantized_decomposed ops are registered before loading INT8 .pt2
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e  # noqa: F401

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WEIGHTS_DIR = Path("weights")
DATA_DIR = Path("data")
COCO128_IMAGES = DATA_DIR / "coco128" / "images" / "train2017"
YOLO_PT2 = WEIGHTS_DIR / "yolov8s.pt2"
YOLO_INT8_PT2 = WEIGHTS_DIR / "yolov8s_int8.pt2"
YOLO_PT = WEIGHTS_DIR / "yolov8s.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval_images(
    image_dir: Path,
    num_images: int = 32,
    image_size: int = 640,
) -> list[torch.Tensor]:
    """Load and preprocess images for evaluation."""
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    paths = sorted(image_dir.glob("*.jpg"))[:num_images]
    if not paths:
        raise FileNotFoundError(f"No .jpg images in {image_dir}")
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(transform(img).unsqueeze(0))
    return tensors


def print_size_of_model(model, label: str) -> float:
    """Save state_dict to a temp file, print its size, and return MB."""
    tmp = "temp_size_check.p"
    torch.save(model.state_dict(), tmp)
    size_mb = os.path.getsize(tmp) / (1024 * 1024)
    os.remove(tmp)
    print(f"  {label}: {size_mb:.2f} MB")
    return size_mb


def collect_outputs(model, images: list[torch.Tensor]) -> list[torch.Tensor]:
    """Run each image through a model and return flattened output tensors."""
    outputs = []
    with torch.no_grad():
        for img in images:
            out = model(img)
            if isinstance(out, (tuple, list)):
                out = out[0]
            outputs.append(out.flatten().float())
    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  YOLOv8s Quantization Evaluation: FP32 vs INT8")
    print("=" * 60)

    for p in [YOLO_PT2, YOLO_INT8_PT2]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run:\n"
                "  python -m compression.prepare_yolov8"
            )

    # ----------------------------------------------------------------
    # 1. Load both models
    # ----------------------------------------------------------------
    print("\n[1/4] Loading models...")

    fp32_ep = torch.export.load(str(YOLO_PT2))
    fp32_model = fp32_ep.module()

    int8_ep = torch.export.load(str(YOLO_INT8_PT2))
    int8_model = int8_ep.module()

    # ----------------------------------------------------------------
    # 2. Size comparison (state_dict method from official tutorial)
    # ----------------------------------------------------------------
    print("\n[2/4] Measuring model sizes...")
    fp32_mb = print_size_of_model(fp32_model, "FP32 model")
    int8_mb = print_size_of_model(int8_model, "INT8 model")
    reduction = (1 - int8_mb / fp32_mb) * 100

    # ----------------------------------------------------------------
    # 3. Run inference on both models
    # ----------------------------------------------------------------
    print("\n[3/4] Loading evaluation images and running inference...")
    images = load_eval_images(COCO128_IMAGES, num_images=32)
    print(f"  Loaded {len(images)} images")

    fp32_outputs = collect_outputs(fp32_model, images)
    print(f"  FP32 inference done ({len(fp32_outputs)} images)")

    int8_outputs = collect_outputs(int8_model, images)
    print(f"  INT8 inference done ({len(int8_outputs)} images)")

    # ----------------------------------------------------------------
    # 4. Compute metrics
    # ----------------------------------------------------------------
    print("\n[4/4] Computing metrics...")
    cos_sims = []
    mses = []
    max_abs_diffs = []

    for fp32_out, int8_out in zip(fp32_outputs, int8_outputs):
        cos_sims.append(
            F.cosine_similarity(fp32_out.unsqueeze(0), int8_out.unsqueeze(0)).item()
        )
        mses.append(F.mse_loss(fp32_out, int8_out).item())
        max_abs_diffs.append((fp32_out - int8_out).abs().max().item())

    cos_mean = sum(cos_sims) / len(cos_sims)
    cos_min = min(cos_sims)
    mse_mean = sum(mses) / len(mses)
    max_abs_diff = max(max_abs_diffs)

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------
    print()
    print("=" * 60)
    print("  SIZE")
    print("=" * 60)
    print(f"  FP32 state_dict:   {fp32_mb:.2f} MB")
    print(f"  INT8 state_dict:   {int8_mb:.2f} MB")
    print(f"  Reduction:         {reduction:+.1f}%")

    print()
    print("=" * 60)
    print(f"  ACCURACY  ({len(images)} COCO128 images)")
    print("=" * 60)
    print(f"  Cosine similarity (mean): {cos_mean:.6f}")
    print(f"  Cosine similarity (min):  {cos_min:.6f}")
    print(f"  MSE (mean):               {mse_mean:.6f}")
    print(f"  Max absolute diff:        {max_abs_diff:.4f}")
    print()

    if cos_mean > 0.999:
        verdict = "EXCELLENT -- outputs near-identical to FP32"
    elif cos_mean > 0.99:
        verdict = "GOOD -- minor numerical drift, unlikely to affect accuracy"
    elif cos_mean > 0.95:
        verdict = "ACCEPTABLE -- some drift, validate on full dataset"
    else:
        verdict = "DEGRADED -- significant loss, needs more calibration data"

    print(f"  Verdict: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
