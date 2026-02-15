# @Time    : 2026-02-14
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : lower_to_pte.py

"""
Lower an INT8 q/dq .pt2 model to an ExecuTorch .pte file via XNNPACK.

This is the final compilation step that:
  - Packs INT8 weights (4x smaller than FP32)
  - Delegates quantized ops to the XNNPACK runtime (optimized for ARM / RPi)
  - Produces a single .pte file ready for on-device inference

Usage:
    python -m compression.lower_to_pte
    python -m compression.lower_to_pte --input weights/custom_int8.pt2 --output weights/custom.pte
"""

import argparse
import os
from pathlib import Path

import torch

# ExecuTorch lowering APIs
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)

# Ensure quantized_decomposed ops are registered
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e  # noqa: F401

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
WEIGHTS_DIR = Path("weights")
DEFAULT_INPUT = WEIGHTS_DIR / "yolov8s_int8.pt2"
DEFAULT_OUTPUT = WEIGHTS_DIR / "yolov8s_int8_xnnpack.pte"


def lower_to_pte(input_path: str, output_path: str) -> Path:
    """Lower a quantized .pt2 (q/dq graph) to an XNNPACK .pte file.

    Args:
        input_path: Path to the INT8 q/dq .pt2 exported program.
        output_path: Where to write the final .pte file.

    Returns:
        Path to the saved .pte file.
    """
    input_p = Path(input_path)
    output_p = Path(output_path)

    if not input_p.exists():
        raise FileNotFoundError(
            f"{input_p} not found. Run the quantization step first:\n"
            "  python -m compression.prepare_yolov8 --step quantize"
        )

    output_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load the quantized exported program
    print(f"Loading quantized model from {input_p} ...")
    ep = torch.export.load(str(input_p))

    # 2. Lower to edge IR and delegate to XNNPACK in one shot
    print("Lowering to edge IR + XNNPACK backend ...")
    et_program = to_edge_transform_and_lower(
        ep,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    ).to_executorch()

    # 3. Write the .pte binary
    print(f"Saving .pte to {output_p} ...")
    with open(output_p, "wb") as f:
        f.write(et_program.buffer)

    # 4. Report sizes
    input_mb = input_p.stat().st_size / (1024 * 1024)
    output_mb = output_p.stat().st_size / (1024 * 1024)

    print()
    print("=" * 55)
    print("  Lowering complete")
    print("=" * 55)
    print(f"  INT8 q/dq .pt2:     {input_mb:.2f} MB  (intermediate)")
    print(f"  XNNPACK .pte:       {output_mb:.2f} MB  (deployable)")
    print(f"  Compression ratio:  {input_mb / output_mb:.1f}x")
    print()

    # Also compare against the original FP32 if it exists
    fp32_pt2 = WEIGHTS_DIR / "yolov8s.pt2"
    if fp32_pt2.exists():
        fp32_mb = fp32_pt2.stat().st_size / (1024 * 1024)
        print(f"  FP32 .pt2 (ref):    {fp32_mb:.2f} MB")
        print(f"  FP32 -> .pte ratio: {fp32_mb / output_mb:.1f}x")
        print()

    print(f"  Ready for Raspberry Pi: {output_p}")
    print("=" * 55)

    return output_p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lower INT8 q/dq .pt2 to XNNPACK .pte for Raspberry Pi"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help=f"Path to quantized .pt2 (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output .pte path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    lower_to_pte(args.input, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
