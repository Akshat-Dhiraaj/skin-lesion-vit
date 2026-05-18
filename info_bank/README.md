# 📍 START HERE — Project Knowledge Base

**New to this project? Read the numbered files in order.** The number prefix *is* the reading path — open `01_…`, then `02_…`, and so on. You do not need to read all of them; stop when you have what you need (see the "stop after" column).

| # | File | What it gives you | Stop after this if you only want… |
|---|------|-------------------|-----------------------------------|
| 01 | [`01_RESULTS.md`](01_RESULTS.md) | The headline numbers + figures + a copy-paste recipe to reproduce them | the 60-second "what was achieved" |
| 02 | [`02_MODELS_AND_TECHNIQUES.md`](02_MODELS_AND_TECHNIQUES.md) | Every model & technique: what it is, how it works, why used, the result, the paper link | to understand or cite any component |
| 03 | [`03_COMPLETE_GUIDE.md`](03_COMPLETE_GUIDE.md) | The 32-part teach-yourself guide (assumes zero ML background → TENT). Parts 1–16 = v1, 17–32 = v2 | the full story, taught from scratch |
| 04 | [`04_FINDINGS_AND_RESUME.md`](04_FINDINGS_AND_RESUME.md) | Findings, negative results, what's publishable, ready-to-paste résumé bullets | to write this up or put it on a CV |
| 05 | [`05_ROADMAP.md`](05_ROADMAP.md) | What is done vs. deliberately deferred | to know what's left to do |
| 06 | [`06_AI_AGENT_CONTEXT.md`](06_AI_AGENT_CONTEXT.md) | Machine-oriented project state: pitfalls, conventions, "source of truth" for an AI/engineer resuming work | to start coding on it |

**Typical paths:**
- *"Just show me what this project is and what it achieved"* → **01**, then skim **02**.
- *"I want to learn everything from the ground up"* → **03** (it links out to 01/02/04 where relevant).
- *"I'm putting this on my résumé / writing it up"* → **01** → **04**.
- *"I'm an engineer (or AI) picking up the work"* → **06** → **05** → the code in [`../v2/`](../v2/).

Historical raw logs (not part of the reading path) live in [`archive/`](archive/): `OVERNIGHT_RESULTS.md` (chronological run log) and `OVERNIGHT_QUEUE.md` (the original task spec).

The project's own landing page is the repo root [`../README.md`](../README.md); the v2 codebase has its own [`../v2/README.md`](../v2/README.md) and phase ledger [`../v2/PHASES.md`](../v2/PHASES.md).

---

## Repository layout

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
    ├── 01_RESULTS.md … 06_AI_AGENT_CONTEXT.md
    └── archive/                 historical run logs (not in the reading path)
```

> **Path note.** Some documents were written when the layout was
> `D:\ML\skin-lesion-vit\` (v1) + `D:\ML\skin-lesion-vit-v2\` (v2) as
> *separate* repos, and when these docs lived at the `D:\ML\` root.
> Bulk path rewrites were applied. If a deep-in-prose path still looks off,
> the mapping is: old `D:\ML\skin-lesion-vit-v2\X` → now
> `D:\ML\skin-lesion-vit\v2\X`; old `D:\ML\<NAME>.md` → now
> `D:\ML\skin-lesion-vit\info_bank\<numbered-name>.md`.
