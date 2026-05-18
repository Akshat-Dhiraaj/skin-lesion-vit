# info_bank — Project Knowledge Base

Cross-cutting documentation for the skin lesion classification project. These files were consolidated here from the old `D:\ML\` root when the two project directories were merged into one git repository.

## Repository layout (post-consolidation)

```
skin-lesion-vit/                <- THE git repo (this repository)
├── README.md                   project landing page (v1 benchmark + pointers)
├── src/ experiments/ ...        v1 — 22-phase image-level benchmark
├── v2/                          v2 — lesion-level honest re-do + fairness audit
│   ├── README.md                v2 project page
│   ├── PHASES.md                v2 phase ledger
│   ├── src/ scripts/ tests/
│   └── figures/                 publication figures (committed)
└── info_bank/                   <- you are here: cross-cutting docs
```

> **Path note.** Some documents below were written when the layout was
> `D:\ML\skin-lesion-vit\` (v1) + `D:\ML\skin-lesion-vit-v2\` (v2) as
> *separate* repos. Bulk path rewrites were applied (`skin-lesion-vit-v2/`
> → `../v2/`, prose `D:\ML\skin-lesion-vit-v2` → `D:\ML\skin-lesion-vit\v2`).
> If a deep-in-prose path still looks off, the mapping is:
> old `D:\ML\skin-lesion-vit-v2\X` → now `D:\ML\skin-lesion-vit\v2\X`;
> old `D:\ML\<doc>.md` → now `D:\ML\skin-lesion-vit\info_bank\<doc>.md`.

## What's in here

| File | Purpose | Read if you want… |
|---|---|---|
| [`Skin_Lesion_Project_Complete_Guide.md`](Skin_Lesion_Project_Complete_Guide.md) | 32-part teach-yourself guide (zero-ML-background → TENT). Parts 1–16 cover v1, 17–32 cover v2. | the full story, pedagogically |
| [`RESULTS_REPORT.md`](RESULTS_REPORT.md) | Auto-generated headline results + embedded figures + recipe. | the 60-second portfolio view |
| [`MODELS.md`](MODELS.md) | Every model & technique: what / how it works / why used / result / paper link. | to understand or cite any component |
| [`FINDINGS.md`](FINDINGS.md) | Findings, negative results, publishability, graded resume bullets. | to write this up or put it on a CV |
| [`AI_context.md`](AI_context.md) | Single source of truth for an AI agent picking up the project. Pitfalls, conventions. | to resume engineering work |
| [`NEXT_STEPS.md`](NEXT_STEPS.md) | What's done vs deliberately deferred (archive index). | to know what's left |
| [`OVERNIGHT_RESULTS.md`](OVERNIGHT_RESULTS.md) | Chronological log of every overnight run. | the raw run history |
| [`OVERNIGHT_QUEUE.md`](OVERNIGHT_QUEUE.md) | The original task queue spec (historical). | how the overnight work was scoped |

## Suggested reading order

1. [`../README.md`](../README.md) — repo landing page
2. [`RESULTS_REPORT.md`](RESULTS_REPORT.md) — what was achieved (with figures)
3. [`MODELS.md`](MODELS.md) — how/why each model & technique
4. [`Skin_Lesion_Project_Complete_Guide.md`](Skin_Lesion_Project_Complete_Guide.md) — depth
5. [`FINDINGS.md`](FINDINGS.md) — publishability + resume framing
6. [`../v2/README.md`](../v2/README.md) — the v2 project page
