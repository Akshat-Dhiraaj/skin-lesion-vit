"""
scripts/leaderboard.py
----------------------
Aggregate all experiments/<phase>/results.txt into one ranked markdown table.

Usage:
    python scripts/leaderboard.py                    # print to stdout
    python scripts/leaderboard.py --output LEADERBOARD.md
"""

import argparse
import re
from pathlib import Path


def parse_results(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    val_auc = re.search(r"Best Val AUC\s*:\s*([\d.]+)", text)
    test_auc = re.search(r"Test AUC\s*:\s*([\d.]+)", text)
    test_acc = re.search(r"Test Acc\s*:\s*([\d.]+)", text)
    if not (test_auc and test_acc):
        return None
    return {
        "name": path.parent.name,
        "val_auc": float(val_auc.group(1)) if val_auc else None,
        "test_auc": float(test_auc.group(1)),
        "test_acc": float(test_acc.group(1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="experiments")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = [
        r for p in sorted(Path(args.exp_dir).glob("*/results.txt"))
        if (r := parse_results(p)) is not None
    ]
    if not results:
        print("No results found.")
        return

    results.sort(key=lambda r: r["test_auc"], reverse=True)

    lines = [
        "# HAM10000 Leaderboard (patient-level splits)\n",
        "| Rank | Model | Val AUC | Test AUC | Test Acc |",
        "|------|-------|---------|----------|----------|",
    ]
    for i, r in enumerate(results, 1):
        val = f"{r['val_auc']:.4f}" if r["val_auc"] is not None else "—"
        lines.append(
            f"| {i} | `{r['name']}` | {val} | **{r['test_auc']:.4f}** | {r['test_acc']:.4f} |"
        )

    out = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()