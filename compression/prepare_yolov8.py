# @Time    : 2026-02-14
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : prepare_yolov8.py

"""
End-to-end test: download YOLOv8s, build calibration data from COCO128,
export to .pt2, and run the INT8 quantization pipeline.

Usage:
    python -m compression.prepare_yolov8          # full pipeline
    python -m compression.prepare_yolov8 --step export   # export only
    python -m compression.prepare_yolov8 --step quantize # quantize only
"""

import argparse
import zipfile
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

from compression.quantize import universal_compress
from compression.utils import save_torch_export

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WEIGHTS_DIR = Path("weights")
DATA_DIR = Path("data")
COCO128_DIR = DATA_DIR / "coco128"
COCO128_IMAGES = COCO128_DIR / "images" / "train2017"
YOLO_PT = WEIGHTS_DIR / "yolov8s.pt"
YOLO_PT2 = WEIGHTS_DIR / "yolov8s.pt2"
YOLO_INT8_PT2 = WEIGHTS_DIR / "yolov8s_int8.pt2"
COCO128_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
)
INPUT_SIZE = (1, 3, 640, 640)  # B, C, H, W


# ---------------------------------------------------------------------------
# Step 1 -- Acquire assets
# ---------------------------------------------------------------------------

def download_yolov8_model() -> Path:
    """Download YOLOv8s via the ultralytics API (auto-caches)."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if YOLO_PT.exists():
        print(f"[skip] {YOLO_PT} already exists")
        return YOLO_PT

    print("Downloading YOLOv8s weights via ultralytics...")
    # YOLO() auto-downloads the checkpoint if not present
    yolo = YOLO("yolov8s.pt")
    # Copy the downloaded file into our weights/ directory
    src = Path(yolo.ckpt_path)
    if src != YOLO_PT:
        import shutil
        shutil.copy2(src, YOLO_PT)
    print(f"YOLOv8s saved to {YOLO_PT}")
    return YOLO_PT


def download_coco128() -> Path:
    """Download and unzip the COCO128 calibration dataset (7 MB, 128 images)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if COCO128_IMAGES.exists() and any(COCO128_IMAGES.iterdir()):
        print(f"[skip] COCO128 already present at {COCO128_DIR}")
        return COCO128_IMAGES

    zip_path = DATA_DIR / "coco128.zip"
    if not zip_path.exists():
        print(f"Downloading COCO128 from {COCO128_URL} ...")
        torch.hub.download_url_to_file(COCO128_URL, str(zip_path))

    print("Extracting COCO128...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()  # clean up zip after extraction

    print(f"COCO128 ready at {COCO128_DIR}")
    return COCO128_IMAGES


# ---------------------------------------------------------------------------
# Step 2 -- Build calibration tensors
# ---------------------------------------------------------------------------

def build_calibration_data(
    image_dir: Path,
    num_images: int = 16,
    image_size: int = 640,
) -> list[torch.Tensor]:
    """Load images from a directory and return a list of preprocessed tensors.

    Each tensor has shape (1, 3, image_size, image_size) with values in [0, 1],
    matching what YOLOv8 expects for inference.
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),  # HWC uint8 -> CHW float [0, 1]
    ])

    image_paths = sorted(image_dir.glob("*.jpg"))[:num_images]
    if not image_paths:
        raise FileNotFoundError(
            f"No .jpg images found in {image_dir}. "
            "Run the download step first."
        )

    tensors: list[torch.Tensor] = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)
        tensors.append(tensor)

    print(f"Built {len(tensors)} calibration tensors of shape {tensors[0].shape}")
    return tensors


# ---------------------------------------------------------------------------
# Step 3 -- Export to .pt2
# ---------------------------------------------------------------------------

def export_yolov8_pt2() -> Path:
    """Load YOLOv8s and export the inner nn.Module to .pt2 via torch.export."""
    if YOLO_PT2.exists():
        print(f"[skip] {YOLO_PT2} already exists")
        return YOLO_PT2

    print(f"Loading YOLOv8s from {YOLO_PT} ...")
    yolo = YOLO(str(YOLO_PT))

    # yolo.model is the raw DetectionModel (nn.Module)
    inner_model = yolo.model
    inner_model.eval()
    inner_model.float()

    example_input = torch.randn(*INPUT_SIZE)
    print("Exporting with torch.export (strict=False) ...")
    save_torch_export(inner_model, example_input, str(YOLO_PT2))
    return YOLO_PT2


# ---------------------------------------------------------------------------
# Step 4 -- Quantize
# ---------------------------------------------------------------------------

def quantize_yolov8(calibration_data: list[torch.Tensor]) -> Path:
    """Run the universal_compress pipeline on the exported .pt2."""
    print(f"\n{'='*50}")
    print("Running INT8 quantization pipeline")
    print(f"{'='*50}\n")
    output = universal_compress(
        model_path=str(YOLO_PT2),
        calibration_data=calibration_data,
        output_path=str(YOLO_INT8_PT2),
    )
    return Path(output)


# ---------------------------------------------------------------------------
# Size comparison helper
# ---------------------------------------------------------------------------

def compare_sizes(original: Path, quantized: Path):
    """Print a before/after size comparison."""
    orig_mb = original.stat().st_size / (1024 * 1024)
    quant_mb = quantized.stat().st_size / (1024 * 1024)
    reduction = (1 - quant_mb / orig_mb) * 100
    print(f"\n{'='*50}")
    print(f"Original  (.pt2): {orig_mb:.2f} MB")
    print(f"Quantized (.pt2): {quant_mb:.2f} MB")
    print(f"File size delta:  {reduction:+.1f}%")
    print()
    print(
        "NOTE: The .pt2 stores a q/dq graph (INT8 weights + scales\n"
        "+ dequantize ops), so file size may not shrink yet.\n"
        "True INT8 size reduction happens when lowered to an\n"
        "edge runtime (e.g. ExecuTorch + XNNPACK on Raspberry Pi)."
    )
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 quantization test")
    parser.add_argument(
        "--step",
        choices=["download", "export", "quantize", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    if args.step in ("download", "all"):
        download_yolov8_model()
        download_coco128()

    if args.step in ("export", "all"):
        export_yolov8_pt2()

    if args.step in ("quantize", "all"):
        cal_data = build_calibration_data(COCO128_IMAGES)
        result = quantize_yolov8(cal_data)
        compare_sizes(YOLO_PT2, result)

    print("\nDone.")


if __name__ == "__main__":
    main()
