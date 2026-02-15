# @Time    : 2026-02-14 18:50
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : utils.py

# Utils for compression tools

from pathlib import Path

from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn


def download_yolov8(local_dir: str = "./weights") -> list[Path]:
    """Download YOLOv8 checkpoints from HuggingFace.

    Returns list of downloaded file paths.
    """
    repo_id = "Ultralytics/YOLOv8"
    files_to_download = ["yolov8s.pt", "yolov8n.pt"]
    paths: list[Path] = []

    for filename in files_to_download:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
        )
        print(f"Downloaded {filename} -> {path}")
        paths.append(Path(path))

    return paths


def save_torch_export(
    model: nn.Module,
    example_input: torch.Tensor,
    output_path: str = "upload_to_compressor.pt2",
) -> str:
    """Export a model with torch.export and save as .pt2.

    Uses export_for_training with strict=False so that models with
    light dynamic control-flow (e.g. YOLO detection heads) can still
    be captured.  The resulting .pt2 file bakes architecture + weights
    into a single portable artifact.

    Args:
        model: Any nn.Module in eval mode.
        example_input: A single representative input tensor
                       (e.g. torch.randn(1, 3, 640, 640) for YOLO).
        output_path: Destination .pt2 file path.

    Returns:
        The output_path string for convenience.
    """
    model.eval()
    with torch.no_grad():
        exported_program = torch.export.export(
            model,
            (example_input,),
            strict=False,
        )
    torch.export.save(exported_program, output_path)
    print(f"Exported model saved to {output_path}")
    return output_path
