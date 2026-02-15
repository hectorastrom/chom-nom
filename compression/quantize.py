# @Time    : 2026-02-14 18:47
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : quantize.py

# Quantize any fp32 .pt2 checkpoint to INT8 via the PT2E flow

import torch
import torch.export
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)


def universal_compress(
    model_path: str,
    calibration_data: list[torch.Tensor],
    output_path: str = "compressed_model.pt2",
):
    """Load a .pt2 exported model, quantize it to INT8 via PT2E, and save.

    Args:
        model_path: Path to a .pt2 file produced by torch.export.save().
        calibration_data: List of input tensors for calibration (~ 10 batches).
        output_path: Where to write the quantized .pt2 file.
    """
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

    # 4. CALIBRATION -- run data so observers record min/max values
    print(f"Calibrating on {min(len(calibration_data), 10)} batches...")
    with torch.no_grad():
        for i, batch in enumerate(calibration_data):
            prepared_model(batch)
            if i >= 9:
                break

    # 5. CONVERT -- swap FP32 ops for INT8 ops and bake in the scales
    print("Converting to INT8...")
    quantized_model = convert_pt2e(prepared_model)

    # 6. SAVE -- re-export then save as .pt2
    print("Re-exporting quantized model...")
    example_input = calibration_data[0]
    quantized_ep = torch.export.export(quantized_model, (example_input,))
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
