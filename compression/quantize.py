# @Time    : 2026-02-14 18:47
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : quantize.py

# Quantize any fp32 .pt2 checkpoint to INT8 via the PT2E flow

from pathlib import Path

import torch
import torch.export
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


def universal_compress(
    model_path: str,
    calibration_dataset: Dataset,
    output_path: str = "compressed_model.pt2",
    batch_size: int = 1,
    num_calibration_batches: int = 10,
):
    """Load a .pt2 exported model, quantize it to INT8 via PT2E, and save.

    Accepts any torch Dataset that yields (x, y) tuples. Only x is used
    for calibration; y is ignored but keeps the interface consistent with
    standard supervised datasets.

    Args:
        model_path: Path to a .pt2 file produced by torch.export.save().
        calibration_dataset: A torch Dataset returning (x, y) per sample.
        output_path: Where to write the quantized .pt2 file.
        batch_size: Batch size for the calibration DataLoader.
        num_calibration_batches: How many batches to run through the model
            for observer calibration (default 10).
    """
    # 0. VALIDATE -- catch the most common mistake early
    model_p = Path(model_path)
    if not model_p.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if model_p.suffix != ".pt2":
        raise ValueError(
            f"Expected a .pt2 exported program, got '{model_p.suffix}' file: "
            f"{model_path}\n"
            "Hint: export your model first with compression.utils.save_torch_export()."
        )

    print(f"Loading {model_path}...")
    # 1. LOAD -- universal load without needing original class definitions
    ep = torch.export.load(model_path)

    print("Inspecting model structure...")
    inspect_model(ep)

    # 2. EXTRACT the GraphModule that prepare_pt2e expects
    model = ep.module()

    # 3. QUANTIZATION (PT2E Flow)
    # XNNPACK is the highly optimized backend for ARM (Raspberry Pi)
    quantizer = XNNPACKQuantizer()
    quantizer.set_global(
        get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=False,
        )
    )

    print("Preparing for quantization...")
    prepared_model = prepare_pt2e(model, quantizer)

    # 4. CALIBRATION -- wrap dataset in a DataLoader and feed x through model
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    num_batches = min(len(calibration_loader), num_calibration_batches)
    print(f"Calibrating on {num_batches} batches (batch_size={batch_size})...")

    used_microbatch_fallback = False
    with torch.no_grad():
        for i, (x, _y) in enumerate(
            tqdm(
                calibration_loader,
                total=num_batches,
                desc="Calibration",
                unit="batch",
            )
        ):
            try:
                prepared_model(x)
            except AssertionError as exc:
                # Many exported programs are guarded on a fixed batch size of 1.
                # Fall back to per-sample calibration so larger loader batches
                # still work without failing the full pipeline.
                if "x.size()[0] == 1" not in str(exc):
                    raise
                used_microbatch_fallback = True
                for j in range(x.shape[0]):
                    prepared_model(x[j:j + 1])
            if i >= num_calibration_batches - 1:
                break

    if used_microbatch_fallback:
        print(
            "Calibration batch guard detected; using per-sample fallback "
            "(effective batch_size=1)."
        )

    # 5. CONVERT -- swap FP32 ops for INT8 ops and bake in the scales
    print("Converting to INT8...")
    quantized_model = convert_pt2e(prepared_model)

    # 6. SAVE -- re-export then save as .pt2
    print("Re-exporting quantized model...")
    example_x, _example_y = calibration_dataset[0]
    if example_x.ndim == 3:
        example_x = example_x.unsqueeze(0)
    example_x = example_x.float()
    quantized_ep = torch.export.export(quantized_model, (example_x,))
    torch.export.save(quantized_ep, output_path)
    print(f"Quantized model saved to {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def inspect_model(ep: torch.export.ExportedProgram):
    """Print a summary of ATen operations found in an ExportedProgram."""
    graph = ep.graph_module.graph
    print("-" * 40)
    print(f"Graph node count: {len(list(graph.nodes))}")

    op_counts: dict[str, int] = {}
    for node in graph.nodes:
        if node.op == "call_function":
            op_name = str(node.target)
            op_counts[op_name] = op_counts.get(op_name, 0) + 1

    print("Detected operations (conv / linear):")
    for op, count in sorted(op_counts.items()):
        if "conv" in op or "linear" in op:
            print(f"  {op}: {count}")
    print("-" * 40)
