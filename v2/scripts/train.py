"""
scripts/train.py
----------------
CLI entry point for all training phases.

Usage:
    python scripts/train.py --model dinov2_registers_ft
    python scripts/train.py --model pandem_probe --num_workers 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import run_phase
from src.models.registry import REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description="Skin lesion training engine v2")
    parser.add_argument(
        "--model", required=True, choices=list(REGISTRY.keys()),
        help="Registry key for the model to train",
    )
    parser.add_argument(
        "--data_dir", default=r"D:\ML\skin-lesion-vit\data\raw",
        help="Path to HAM10000_metadata.csv and image folders",
    )
    parser.add_argument("--ckpt_base", default="experiments")
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader workers (use 0 on Windows if fork errors occur)",
    )
    args = parser.parse_args()

    result = run_phase(
        model_key=args.model,
        data_dir=args.data_dir,
        ckpt_base=args.ckpt_base,
        num_workers=args.num_workers,
    )
    print(f"\nFinal — Test AUC: {result['test_auc']:.4f} | Test Acc: {result['test_acc']:.4f}")


if __name__ == "__main__":
    main()