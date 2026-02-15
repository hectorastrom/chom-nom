# @Time    : 2026-02-14 18:50
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : utils.py

# Utils for compression tools

from pathlib import Path

from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset persistence
# ---------------------------------------------------------------------------

def save_dataset(dataset: Dataset, output_path: str) -> str:
    """Serialize any (x, y) Dataset to a single .pt file on disk.

    Iterates the full dataset, stacks all x tensors into one tensor and
    all y values into another, then writes them with torch.save.  The
    resulting file can be loaded back with SavedDataset.

    Args:
        dataset: Any torch Dataset whose __getitem__ returns (x, y).
        output_path: Destination .pt file path.

    Returns:
        The output_path string for convenience.
    """
    xs: list[torch.Tensor] = []
    ys: list = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        xs.append(x)
        ys.append(y)

    x_tensor = torch.stack(xs)

    # y may already be tensors (e.g. one-hot labels) or plain scalars
    if isinstance(ys[0], torch.Tensor):
        y_tensor = torch.stack(ys)
    else:
        y_tensor = torch.tensor(ys)

    torch.save({"x": x_tensor, "y": y_tensor}, output_path)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(
        f"Saved dataset ({len(xs)} samples, {size_mb:.1f} MB) -> {output_path}"
    )
    return output_path


class SavedDataset(Dataset):
    """Dataset loaded from a .pt file produced by save_dataset().

    Each sample is an (x, y) tuple where x and y are tensors.
    """

    def __init__(self, path: str):
        data = torch.load(path, weights_only=True)
        self.x: torch.Tensor = data["x"]
        self.y: torch.Tensor = data["y"]
        print(f"Loaded dataset: {len(self)} samples, x shape {self.x.shape}")

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------------------
# HuggingFace download helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# torch.export helper
# ---------------------------------------------------------------------------

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
