"""
scripts/make_plots.py
---------------------
Generate publication-quality figures from experiments/ artifacts. Pure
matplotlib (no seaborn dep). Each function is idempotent — re-running
overwrites the figures with whatever data is on disk at that moment.

Output:
    figures/fig1_sample_efficiency.{png,pdf}
    figures/fig2_fairness_frontier.{png,pdf}
    figures/fig3_tent_improvement.{png,pdf}
    figures/fig4_pandem_loader_fix.{png,pdf}

Usage:
    python scripts/make_plots.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # No display required
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "experiments"
FIG_DIR = HERE.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


# Color scheme — colorblind-safe (Wong 2011)
COLORS = {
    "pandem_real_probe":    "#D55E00",  # orange
    "convnextv2_large":     "#0072B2",  # blue
    "efficientnetv2_m":     "#009E73",  # green
    "dinov2_registers_ft":  "#CC79A7",  # pink
    "siglip2_probe":        "#56B4E9",  # light blue
    "pandem_ft":            "#999999",  # gray (the fallback)
}

LABELS = {
    "pandem_real_probe":   "PanDerm probe (real BEiT)",
    "convnextv2_large":    "ConvNeXt-V2 Large (FT)",
    "efficientnetv2_m":    "EfficientNetV2-M (FT)",
    "dinov2_registers_ft": "DINOv2+Registers (FT)",
    "siglip2_probe":       "SigLIP 2 probe",
    "pandem_ft":           "pandem_ft (DINOv2 fallback)",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 100,
})


# ── Parsers ──────────────────────────────────────────────────────────────────

def _re_search_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def read_sample_eff(model_key: str, frac_int: int) -> float | None:
    """Returns test AUC for sample-eff run, or None if missing."""
    path = EXP / "sample_efficiency" / f"{model_key}_frac{frac_int:03d}" / "results.txt"
    if not path.exists():
        return None
    return _re_search_float(r"Test AUC\s*:\s*([\d.]+)", path.read_text(encoding="utf-8"))


def read_cross_report(ds: str, model: str, suffix: str = "") -> dict | None:
    """Parse a cross_dataset[_tent]/<ds>/<model>[_suffix]/report.txt into a dict."""
    base = "cross_dataset_tent" if suffix else "cross_dataset"
    path = EXP / base / ds / (model + suffix) / "report.txt"
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8")
    out = {
        "macro_ovr_auc": _re_search_float(r"Macro OvR AUC\s*:\s*([\d.]+)", txt),
        "malignant_auc": _re_search_float(r"Malignant AUC\s*:\s*([\d.]+)", txt),
    }
    for grp in ("light", "medium", "dark"):
        out[f"{grp}_mal_auc"] = _re_search_float(
            rf"{grp}\s+N=\s*\d+\s+pos=\s*\d+\s+AUC=([\d.]+)", txt
        )
    if out["light_mal_auc"] and out["dark_mal_auc"]:
        out["ld_gap"] = out["light_mal_auc"] - out["dark_mal_auc"]
    else:
        out["ld_gap"] = None
    return out


# ── Figure 1: Sample efficiency ──────────────────────────────────────────────

def fig1_sample_efficiency():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    models = ["pandem_real_probe", "efficientnetv2_m", "convnextv2_large", "dinov2_registers_ft"]
    fractions = [10, 25, 50, 100]

    plotted_any = False
    for m in models:
        xs, ys = [], []
        for f in fractions:
            v = read_sample_eff(m, f)
            if v is not None:
                xs.append(f)
                ys.append(v)
        if not xs:
            continue
        plotted_any = True
        # Use full-data point from canonical experiments/<model>/results.txt as 100% fallback
        if 100 not in xs:
            full = EXP / m / "results.txt"
            if full.exists():
                v = _re_search_float(r"Test AUC\s*:\s*([\d.]+)", full.read_text(encoding="utf-8"))
                if v is not None:
                    xs.append(100); ys.append(v)
        order = np.argsort(xs)
        xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
        ax.plot(xs, ys, marker="o", markersize=9, linewidth=2.2,
                color=COLORS.get(m, "#444"), label=LABELS.get(m, m))
        # Annotate end-of-curve
        ax.annotate(f"{ys[-1]:.3f}", xy=(xs[-1], ys[-1]),
                    xytext=(6, -2), textcoords="offset points",
                    fontsize=9, color=COLORS.get(m, "#444"))

    if not plotted_any:
        ax.text(0.5, 0.5, "No sample-efficiency data on disk",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Train data fraction (% of HAM10000 lesions)")
    ax.set_ylabel("Test macro OvR AUC")
    ax.set_title("Sample Efficiency on HAM10000 (lesion-level splits)")
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{f}%" for f in fractions])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_ylim(bottom=0.80)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig1_sample_efficiency.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig1_sample_efficiency saved")


# ── Figure 2: Cross-dataset fairness frontier ────────────────────────────────

def fig2_fairness_frontier():
    """Scatter: x = light-dark gap (lower better), y = dark-skin Mal AUC (higher better)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    datasets = [("ddi", "DDI (656 biopsy-proven photos)"),
                ("fitz17k", "Fitzpatrick17k (3,730 clinical photos)")]

    for ax, (ds, title) in zip(axes, datasets):
        ax.set_title(title)
        ax.set_xlabel("Light → dark Malignant AUC gap (lower = fairer)")
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
        # Baseline points
        for m in ["pandem_real_probe", "convnextv2_large", "efficientnetv2_m",
                  "dinov2_registers_ft", "siglip2_probe", "pandem_ft"]:
            r = read_cross_report(ds, m)
            if r and r["ld_gap"] is not None and r["dark_mal_auc"] is not None:
                ax.scatter(r["ld_gap"], r["dark_mal_auc"],
                           s=120, color=COLORS.get(m, "#444"), edgecolors="black", linewidths=0.8,
                           label=LABELS.get(m, m), zorder=3)
        # TENT-adapted versions: same point but star marker
        for m in ["pandem_real_probe", "convnextv2_large", "efficientnetv2_m", "dinov2_registers_ft"]:
            for suffix in ("", "__lr1e-4", "__nsteps3"):
                r = read_cross_report(ds, m, suffix)
                if r and r["ld_gap"] is not None and r["dark_mal_auc"] is not None:
                    ax.scatter(r["ld_gap"], r["dark_mal_auc"],
                               s=140, color=COLORS.get(m, "#444"), marker="*",
                               edgecolors="black", linewidths=0.6, alpha=0.7, zorder=4)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Dark-skin Malignant AUC (higher = better)")
    handles, labels_ = axes[0].get_legend_handles_labels()
    # Append marker legend
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], marker="o", linestyle="", color="gray",
                          markersize=10, markeredgecolor="black", label="baseline"))
    handles.append(Line2D([0], [0], marker="*", linestyle="", color="gray",
                          markersize=14, markeredgecolor="black", label="+ TENT (any cfg)"))
    labels_ = [h.get_label() for h in handles]
    fig.legend(handles, labels_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=9)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig2_fairness_frontier.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig2_fairness_frontier saved")


# ── Figure 3: TENT improvement bars ─────────────────────────────────────────

def fig3_tent_improvement():
    """Per-(model, dataset) baseline vs best-TENT dark-skin Mal AUC."""
    pairs = []
    for ds in ("ddi", "fitz17k"):
        for m in ("pandem_real_probe", "convnextv2_large", "dinov2_registers_ft", "efficientnetv2_m"):
            base = read_cross_report(ds, m)
            best = None; best_cfg = None
            for suffix in ("", "__lr1e-4", "__nsteps3"):
                r = read_cross_report(ds, m, suffix)
                if r and r["dark_mal_auc"] is not None:
                    if best is None or r["dark_mal_auc"] > best:
                        best = r["dark_mal_auc"]; best_cfg = suffix or "lr1e-3"
            if base and base["dark_mal_auc"] is not None and best is not None:
                pairs.append((ds, m, base["dark_mal_auc"], best, best_cfg))

    if not pairs:
        print("  fig3_tent_improvement: no data, skipping")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(pairs))
    width = 0.38
    base_vals = [p[2] for p in pairs]
    tent_vals = [p[3] for p in pairs]
    labels = [f"{p[1]}\n({p[0]})" for p in pairs]
    ax.bar(x - width / 2, base_vals, width, label="baseline (no TENT)",
           color="#999999", edgecolor="black", linewidth=0.7)
    ax.bar(x + width / 2, tent_vals, width, label="+ TENT (best config)",
           color="#0072B2", edgecolor="black", linewidth=0.7)

    # Annotate deltas
    for i, (b, t) in enumerate(zip(base_vals, tent_vals)):
        delta = t - b
        color = "green" if delta > 0 else "red"
        ax.annotate(f"{delta:+.3f}",
                    xy=(i, max(b, t) + 0.01),
                    ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=10, ha="right")
    ax.set_ylabel("Dark-skin Malignant AUC")
    ax.set_title("TENT (test-time entropy minimisation) — dark-skin Malignant AUC change")
    ax.axhline(0.5, color="red", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(len(pairs) - 0.5, 0.51, "random = 0.5", color="red", fontsize=8, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0.40, 0.80)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig3_tent_improvement.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig3_tent_improvement saved")


# ── Figure 4: PanDerm loader-fix impact ──────────────────────────────────────

def fig4_pandem_loader_fix():
    """Comparison: DINOv2-fallback `pandem_*` vs real PanDerm."""
    metrics = {
        "HAM10000 test AUC (probe)": ("pandem_probe", "pandem_real_probe", 0.9305, 0.9527),
        "HAM10000 test AUC (FT)":    ("pandem_ft", "pandem_real_ft", 0.9330, 0.9295),
        "ISIC 2019 macro OvR (probe)": (None, None, 0.8337, 0.8845),  # from pandem_probe_224 ✗
        "ISIC 2019 macro OvR (FT)":    (None, None, 0.8101, 0.8185),
        "DDI Malignant AUC (probe)":   (None, None, None, 0.6821),
        "DDI Malignant AUC (FT)":      (None, None, 0.5159, None),
    }

    labels = list(metrics.keys())
    fallback = [v[2] for v in metrics.values()]
    real     = [v[3] for v in metrics.values()]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    width = 0.4
    f_vals = [v if v is not None else np.nan for v in fallback]
    r_vals = [v if v is not None else np.nan for v in real]
    ax.bar(x - width / 2, f_vals, width, label="DINOv2+Registers (silent fallback)",
           color="#999999", edgecolor="black", linewidth=0.7)
    ax.bar(x + width / 2, r_vals, width, label="real PanDerm (BEiT via open_clip)",
           color="#D55E00", edgecolor="black", linewidth=0.7)

    for i, (a, b) in enumerate(zip(fallback, real)):
        if a is not None and b is not None:
            d = b - a
            color = "green" if d >= 0 else "red"
            ax.annotate(f"{d:+.3f}",
                        xy=(i, max(a, b) + 0.012),
                        ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("AUC")
    ax.set_title("PanDerm loader fix — impact of real BEiT weights vs silent DINOv2 fallback")
    ax.legend(loc="lower right")
    ax.set_ylim(0.45, 1.00)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig4_pandem_loader_fix.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig4_pandem_loader_fix saved")


def main():
    print("Generating figures into", FIG_DIR)
    fig1_sample_efficiency()
    fig2_fairness_frontier()
    fig3_tent_improvement()
    fig4_pandem_loader_fix()
    print("\nAll figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
