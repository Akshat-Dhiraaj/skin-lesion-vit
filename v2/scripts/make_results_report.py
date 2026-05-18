"""
scripts/make_results_report.py
------------------------------
Compile a single self-contained markdown report from the live `experiments/`
artifacts. Embeds the four figures from make_plots.py.

Output: D:/ML/RESULTS_REPORT.md
"""
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                         # .../skin-lesion-vit/v2
REPO = ROOT.parent                         # .../skin-lesion-vit  (repo root)
EXP = ROOT / "experiments"                 # v2/experiments
FIG_DIR = ROOT / "figures"                 # v2/figures
OUT = REPO / "info_bank" / "RESULTS_REPORT.md"   # canonical post-consolidation location


def grep_float(pattern: str, path: Path) -> float | None:
    if not path.exists():
        return None
    m = re.search(pattern, path.read_text(encoding="utf-8"))
    return float(m.group(1)) if m else None


def sample_eff(model: str, frac: int) -> float | None:
    return grep_float(r"Test AUC\s*:\s*([\d.]+)",
                      EXP / "sample_efficiency" / f"{model}_frac{frac:03d}" / "results.txt")


def fallback_full(model: str) -> float | None:
    """100% data falls back to the canonical experiments/<model>/results.txt."""
    return grep_float(r"Test AUC\s*:\s*([\d.]+)", EXP / model / "results.txt")


def cross_report(ds: str, model: str, suffix: str = "") -> dict:
    base = "cross_dataset_tent" if suffix else "cross_dataset"
    path = EXP / base / ds / (model + suffix) / "report.txt"
    if not path.exists():
        return {}
    out = {
        "macro": grep_float(r"Macro OvR AUC\s*:\s*([\d.]+)", path),
        "mal":   grep_float(r"Malignant AUC\s*:\s*([\d.]+)", path),
    }
    for grp in ("light", "medium", "dark"):
        out[grp] = grep_float(rf"{grp}\s+N=\s*\d+\s+pos=\s*\d+\s+AUC=([\d.]+)", path)
    if out["light"] is not None and out["dark"] is not None:
        out["gap"] = out["light"] - out["dark"]
    return out


def fmt(v, digits=4) -> str:
    return f"{v:.{digits}f}" if v is not None else "—"


def posthoc_best_auc() -> float | None:
    """Final post-hoc test AUC (the [step 3] / Final line)."""
    p = EXP / "post_hoc" / "results.txt"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8")
    m = re.search(r"Final test AUC\s*:\s*([\d.]+)", txt)
    if m:
        return float(m.group(1))
    m = re.search(r"\[step 3\][^\n]+AUC=([\d.]+)", txt)
    return float(m.group(1)) if m else None


def ddi_tent_ensemble_dark_mal() -> float | None:
    """DDI 'Tuned for fairness' dark-skin Mal AUC from the TENT-aware B4 report."""
    for name in ("cross_dataset_ensemble_tent_best", "cross_dataset_ensemble_tent_lr1e-3"):
        p = EXP / name / "report.txt"
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        ddi = txt.split("FITZ17K")[0]  # isolate the DDI block
        for header in ("Tuned for fairness", "Tuned for accuracy"):
            m = re.search(rf"{header}.*?dark_mal=([\d.]+)", ddi, re.DOTALL)
            if m:
                return float(m.group(1))
    return None


def section_headline():
    return f"""# Skin Lesion v2 — Results Report

> Auto-generated from `experiments/` artifacts on {datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}.

This document captures the headline numbers, figures, and recipe from the v2 (lesion-level-splits, cross-dataset audited) iteration of the skin lesion classification project. For project context, see [`Skin_Lesion_Project_Complete_Guide.md`](Skin_Lesion_Project_Complete_Guide.md); for findings + publishability discussion, see [`FINDINGS.md`](FINDINGS.md).

## TL;DR

| Metric | Value | Where |
|---|---:|---|
| HAM10000 test macro OvR AUC (best ensemble) | **{fmt(posthoc_best_auc())}** | `experiments/post_hoc/results.txt` |
| HAM10000 best single model (ConvNeXt-V2) | {fmt(fallback_full("convnextv2_large"))} | `experiments/convnextv2_large/results.txt` |
| ISIC 2019 cross-dataset best | {fmt(cross_report("isic2019", "pandem_real_probe").get("macro"))} | `pandem_real_probe` (probe-only) |
| DDI dark-skin Malignant AUC, TENT-aware SLSQP ensemble | {fmt(ddi_tent_ensemble_dark_mal())} | `experiments/cross_dataset_ensemble_tent_*/report.txt` |
| Sample-efficiency advantage (PanDerm probe @ 25% data vs EfficientNetV2 @ 100%) | ≈ parity | `experiments/sample_efficiency/` |

---
"""


def section_sample_eff():
    rows = []
    rows.append("## 1. Sample efficiency on HAM10000\n")
    rows.append("![Sample efficiency](../v2/figures/fig1_sample_efficiency.png)\n")
    rows.append("\n| Train fraction | Train lesions | PanDerm probe | EfficientNetV2-M | ConvNeXt-V2 | DINOv2+Reg FT |")
    rows.append("|---|---:|---:|---:|---:|---:|")
    for frac_int, lesions in [(10, 534), (25, 1335), (50, 2668), (100, 5336)]:
        pdp = sample_eff("pandem_real_probe", frac_int) or (fallback_full("pandem_real_probe") if frac_int == 100 else None)
        eff = sample_eff("efficientnetv2_m", frac_int) or (fallback_full("efficientnetv2_m") if frac_int == 100 else None)
        cnx = sample_eff("convnextv2_large", frac_int) or (fallback_full("convnextv2_large") if frac_int == 100 else None)
        din = sample_eff("dinov2_registers_ft", frac_int) or (fallback_full("dinov2_registers_ft") if frac_int == 100 else None)
        rows.append(f"| {frac_int}% | ~{lesions:,} | {fmt(pdp)} | {fmt(eff)} | {fmt(cnx)} | {fmt(din)} |")
    rows.append("")
    rows.append("**Reading the table.** PanDerm's dermatology pretraining means the linear probe saturates early — only the head is being trained on subsampled data, and the head's task is easy when the features are already discriminative. The ImageNet-pretrained CNNs are still improving meaningfully with each data point.\n")
    return "\n".join(rows)


def section_in_distribution():
    rows = []
    rows.append("## 2. In-distribution HAM10000 (lesion-level splits, test N=1,431 / 1,067 lesions)\n")
    rows.append("\n### Single models\n")
    rows.append("| Model | Best Val AUC | Test AUC | TTA Test AUC |")
    rows.append("|---|---:|---:|---:|")
    for model in ["convnextv2_large", "pandem_real_probe", "efficientnetv2_m", "siglip2_probe",
                  "dinov2_registers_ft", "pandem_real_ft"]:
        val = grep_float(r"Best Val AUC\s*:\s*([\d.]+)", EXP / model / "results.txt")
        test = fallback_full(model)
        tta = grep_float(r"\[TEST\][^[]*TTA AUC\s*:\s*([\d.]+)", EXP / model / "tta_results.txt")
        rows.append(f"| `{model}` | {fmt(val)} | {fmt(test)} | {fmt(tta)} |")
    rows.append("")
    rows.append("\n### 4-model post-hoc pipeline (canonical with `pandem_real_probe`)\n")
    ph = EXP / "post_hoc" / "results.txt"
    if ph.exists():
        txt = ph.read_text(encoding="utf-8")
        rows.append("| Stage | Test AUC | Mel Recall |")
        rows.append("|---|---:|---:|")
        for stage_re, name in [
            (r"\[baseline\][^\n]+AUC=([\d.]+)[^\n]+MelRecall=([\d.]+)", "mean ensemble"),
            (r"\[step 1\][^\n]+AUC=([\d.]+)[^\n]+MelRecall=([\d.]+)", "+ per-class SLSQP"),
            (r"\[step 2\][^\n]+AUC=([\d.]+)[^\n]+MelRecall=([\d.]+)", "+ Platt calibration"),
            (r"\[step 3\][^\n]+AUC=([\d.]+)[^\n]+MelRecall=([\d.]+)", "+ F2 thresholds"),
        ]:
            m = re.search(stage_re, txt)
            if m:
                rows.append(f"| {name} | {m.group(1)} | {m.group(2)} |")
    rows.append("")
    return "\n".join(rows)


def section_cross_dataset():
    rows = []
    rows.append("## 3. Cross-dataset evaluation\n")
    rows.append("![Fairness frontier](../v2/figures/fig2_fairness_frontier.png)\n")
    rows.append("\n### ISIC 2019 (HAM10K-7 taxonomy)\n")
    rows.append("| Model | Macro OvR AUC |")
    rows.append("|---|---:|")
    for m in ["pandem_real_probe", "convnextv2_large", "pandem_probe_224",
              "efficientnetv2_m", "pandem_real_ft", "pandem_ft",
              "dinov2_registers_ft", "siglip2_probe"]:
        v = cross_report("isic2019", m).get("macro")
        if v is not None:
            tag = " ✗" if m in ("pandem_probe", "pandem_ft", "pandem_probe_224", "pandem_ft_224") else ""
            rows.append(f"| `{m}`{tag} | {fmt(v)} |")
    rows.append("")
    rows.append("`✗` = silently DINOv2+Registers fallback (see PanDerm loader fix).\n")

    for ds, label in [("ddi", "DDI (Stanford AIMI, 656 biopsy-proven photos)"),
                      ("fitz17k", "Fitzpatrick17k (3,730 clinical photos)")]:
        rows.append(f"\n### {label}\n")
        rows.append("| Model | Macro OvR | Mal AUC | Light Mal | Medium Mal | Dark Mal | L–D Gap |")
        rows.append("|---|---:|---:|---:|---:|---:|---:|")
        for m in ["pandem_real_probe", "convnextv2_large", "efficientnetv2_m",
                  "dinov2_registers_ft", "siglip2_probe", "pandem_ft"]:
            r = cross_report(ds, m)
            if not r:
                continue
            tag = " ✗" if m == "pandem_ft" else ""
            rows.append(
                f"| `{m}`{tag} | {fmt(r.get('macro'))} | {fmt(r.get('mal'))} "
                f"| {fmt(r.get('light'))} | {fmt(r.get('medium'))} | {fmt(r.get('dark'))} "
                f"| {fmt(r.get('gap'), 3)} |"
            )
        rows.append("")
    return "\n".join(rows)


def section_tent():
    rows = []
    rows.append("## 4. Test-time adaptation (TENT)\n")
    rows.append("![TENT improvement](../v2/figures/fig3_tent_improvement.png)\n")
    rows.append("\nPer-(model, dataset) TENT improvement on dark-skin Malignant AUC, comparing baseline (no TENT) against the best-tuned TENT config:\n")
    rows.append("\n| Dataset | Model | Baseline | + TENT (best) | Δ | Best config |")
    rows.append("|---|---|---:|---:|---:|---|")
    for ds in ("ddi", "fitz17k"):
        for m in ("pandem_real_probe", "convnextv2_large", "dinov2_registers_ft", "efficientnetv2_m"):
            base = cross_report(ds, m).get("dark")
            best_val, best_cfg = None, None
            for suffix in ("", "__lr1e-4", "__nsteps3", "__best"):
                d = cross_report(ds, m, suffix).get("dark")
                if d is not None and (best_val is None or d > best_val):
                    best_val = d
                    best_cfg = suffix or "lr1e-3"
            if base is not None and best_val is not None:
                delta = best_val - base
                rows.append(f"| {ds} | `{m}` | {fmt(base)} | **{fmt(best_val)}** | {delta:+.4f} | `{best_cfg}` |")
    rows.append("")
    return "\n".join(rows)


def section_pandem_fix():
    rows = []
    rows.append("## 5. PanDerm loader fix — impact\n")
    rows.append("![PanDerm fix](../v2/figures/fig4_pandem_loader_fix.png)\n")
    rows.append("\nThe DermLIP/PanDerm HuggingFace repo (`redlessone/DermLIP_PanDerm-base-w-PubMed-256`) is an **open_clip** checkpoint without a `config.json`. `transformers.AutoModel.from_pretrained` raises and a silent `try/except` falls back to `facebook/dinov2-with-registers-base`. Every `pandem_*` checkpoint (without `_real`) is therefore DINOv2+Registers, not PanDerm.\n")
    rows.append("\nThe fix (`RealPanDermClassifier` in `src/models/registry.py`): build `timm.models.beit.Beit` with `use_abs_pos_emb=True, use_rel_pos_bias=False, init_values=0.1, global_pool='token'`, load the visual sub-state-dict from `open_clip_pytorch_model.bin` — 0 missing / 0 unexpected keys.\n")
    return "\n".join(rows)


def section_methodology():
    return """## 6. Methodology

- **Honest evaluation**: lesion-level `StratifiedGroupKFold` splits (`src/data/splits.py`) with `assert_no_group_leakage()` enforced in pytest. Same model on the same data: v1 image-level split test AUC 0.9880 → v2 lesion-level split test AUC 0.9660. The 0.022 gap is the leakage inflation v1 had.
- **8-view TTA**: original + H-flip + V-flip + rot 90/180/270 + HV-flip + slight zoom. Softmax averaged.
- **3-stage post-hoc**: per-class SLSQP weights → Platt sigmoid calibration → F2-optimal thresholds. Fit on val, applied to test.
- **Cross-dataset taxonomies**: universal-3 `[malignant, benign, non_neoplastic]` for DDI / Fitz17k; HAM10K-7 for ISIC 2019 (with SCC → akiec). `map_ham10k_probs_to_universal()` projects 7-way → 3-way.
- **Fitzpatrick bucketing**: types 1–2 light, 3–4 medium, 5–6 dark.
- **TENT**: freeze everything except normalisation-layer affines; minimise softmax entropy per test batch via one Adam step; emit adapted softmax.
- **Per-target SLSQP ensemble**: SLSQP weights fit on the target dataset's probabilities (with or without TENT) — quantifies the deployment-specific accuracy-fairness tradeoff.

## 7. Recipe to reproduce headline numbers

```powershell
cd D:\\ML\\skin-lesion-vit-v2
.\\.venv\\Scripts\\Activate.ps1
$env:PYTHONIOENCODING="utf-8"

# In-distribution headline (0.97 macro OvR AUC)
python -m src.post_hoc --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_real_probe

# ISIC 2019 cross-dataset winner
Get-Content experiments\\cross_dataset\\isic2019\\pandem_real_probe\\report.txt

# DDI Fitzpatrick gap closure
python -m src.cross_dataset_ensemble `
    --models convnextv2_large efficientnetv2_m dinov2_registers_ft pandem_real_probe `
    --datasets ddi --probs_subdir cross_dataset_tent --out_suffix _tent_lr1e-3
Get-Content experiments\\cross_dataset_ensemble_tent_lr1e-3\\report.txt
```

## 8. Negative results

- 5-model ensemble (+ siglip2_probe): SLSQP gives the floor weight; test AUC drops to 0.9703. Artifact: `experiments/ensemble/results_5model.txt`.
- Full FT of real PanDerm: `pandem_real_ft` test AUC 0.9295 < probe 0.9527. Pretrained features too task-relevant to disturb on 7k images.
- Binary mel specialist (OA1): test mel AUC 0.8942 but R@P=0.80 only 0.220 — the 7-way model with F2-tuned mel threshold (mel recall 0.566 from post_hoc) beats it.
- TENT on `pandem_real_probe` for Fitz17k: every (lr, n_steps) drops dark-skin Mal AUC below baseline.

## 9. Open work

- Multimodal head (age + sex + anatomical site).
- Grad-CAM / attention visualisation for clinical interpretability.
- Domain generalisation training (GroupDRO, DANN).
- Dermatologist-in-the-loop evaluation.
- Formal write-up / paper submission.

---

*Report assembled from artifacts under `skin-lesion-vit/v2/experiments/` and figures under `skin-lesion-vit/v2/figures/`. Trust the underlying `.txt` files if any number here disagrees with them.*
"""


def main():
    sections = [
        section_headline(),
        section_sample_eff(),
        section_in_distribution(),
        section_cross_dataset(),
        section_tent(),
        section_pandem_fix(),
        section_methodology(),
    ]
    OUT.write_text("\n".join(sections), encoding="utf-8")
    print(f"Report written: {OUT}")


if __name__ == "__main__":
    main()
