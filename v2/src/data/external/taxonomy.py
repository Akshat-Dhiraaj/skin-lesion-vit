"""
src/data/external/taxonomy.py
-----------------------------
Universal label taxonomy and per-dataset mappings for cross-dataset evaluation.

Two levels of taxonomy:

1. HAM10000 7-class scheme (CLASS_NAMES from ham10000.py):
   ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
   - Used when external dataset has a compatible class scheme (ISIC 2019).

2. Universal 3-way scheme — for fundamentally different label schemas:
   - malignant     : life-threatening (melanoma, BCC, SCC, akiec)
   - benign        : non-life-threatening (nevus, BKL, DF, vascular)
   - non_neoplastic: inflammatory/infectious/other (DDI/Fitz17k native)

Decisions documented per-dataset in this file. Every mapping decision is
explicit and reviewable — no silent classes get dropped.
"""

from __future__ import annotations

# ── HAM10000 native classes (the reference) ──────────────────────────────────

HAM10K_CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
HAM10K_TO_IDX = {c: i for i, c in enumerate(HAM10K_CLASSES)}

# Which HAM10000 classes are malignant (clinical relevance)
HAM10K_MALIGNANT = {"mel", "bcc", "akiec"}
HAM10K_BENIGN = {"nv", "bkl", "df", "vasc"}


# ── Universal 3-way taxonomy ──────────────────────────────────────────────────

UNIVERSAL_CLASSES = ["malignant", "benign", "non_neoplastic"]
UNIVERSAL_TO_IDX = {c: i for i, c in enumerate(UNIVERSAL_CLASSES)}

# HAM10000 -> universal
HAM10K_TO_UNIVERSAL = {
    "mel":   "malignant",
    "bcc":   "malignant",
    "akiec": "malignant",
    "nv":    "benign",
    "bkl":   "benign",
    "df":    "benign",
    "vasc":  "benign",
}


# ── ISIC 2019 mapping ────────────────────────────────────────────────────────
# ISIC 2019 has the same 7 HAM10000 classes plus SCC.
# AK in ISIC 2019 == akiec in HAM10000 (actinic keratosis).
# SCC (squamous cell carcinoma) is malignant; merged with akiec for HAM10000
# 7-class comparison, but kept separate when reporting native 8-class metrics.

ISIC2019_NATIVE_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

ISIC2019_TO_HAM10K = {
    "MEL":  "mel",
    "NV":   "nv",
    "BCC":  "bcc",
    "AK":   "akiec",
    "BKL":  "bkl",
    "DF":   "df",
    "VASC": "vasc",
    "SCC":  "akiec",  # merge SCC -> akiec for HAM10000 7-class eval
}

ISIC2019_TO_UNIVERSAL = {
    "MEL":  "malignant",
    "NV":   "benign",
    "BCC":  "malignant",
    "AK":   "malignant",
    "BKL":  "benign",
    "DF":   "benign",
    "VASC": "benign",
    "SCC":  "malignant",
}


# ── DDI (Stanford Diverse Dermatology Images) mapping ────────────────────────
# DDI provides binary malignant (T/F) labels and specific disease names.
# 78 unique disease names — mostly inflammatory/non-neoplastic. We use the
# malignant boolean for the universal mapping. Disease-level mapping to
# HAM10000 7-class is only attempted for the diseases that overlap clearly.

# Diseases in DDI that map clearly to HAM10000 classes.
# Anything not in this dict is treated as "non_neoplastic" universally.
# Source: DDI dataset README + manual review.
DDI_DISEASE_TO_HAM10K = {
    # Melanoma variants
    "melanoma":                                          "mel",
    "melanoma-in-situ":                                  "mel",
    "melanoma-acral-lentiginous":                        "mel",
    "nodular-melanoma-(nm)":                             "mel",
    "lentigo-maligna-melanoma":                          "mel",
    # Nevi
    "nevus":                                             "nv",
    "dysplastic-nevus":                                  "nv",
    "blue-nevus":                                        "nv",
    "atypical-spindle-cell-nevus-of-reed":               "nv",
    # BCC
    "basal-cell-carcinoma":                              "bcc",
    "basal-cell-carcinoma-superficial":                  "bcc",
    "basal-cell-carcinoma-nodular":                      "bcc",
    # Actinic keratosis / SCC -> akiec
    "actinic-keratosis":                                 "akiec",
    "squamous-cell-carcinoma":                           "akiec",
    "squamous-cell-carcinoma-in-situ":                   "akiec",
    "squamous-cell-carcinoma-keratoacanthoma":           "akiec",
    # Seborrheic keratosis -> bkl
    "seborrheic-keratosis":                              "bkl",
    "seborrheic-keratosis-irritated":                    "bkl",
    # Dermatofibroma
    "dermatofibroma":                                    "df",
    # Vascular
    "pyogenic-granuloma":                                "vasc",
    "hemangioma":                                        "vasc",
    "angiokeratoma":                                     "vasc",
    "kaposi-sarcoma":                                    "vasc",
}


def ddi_to_universal(malignant_flag: bool, disease: str) -> str:
    """
    Map a DDI sample to a universal-taxonomy label.
    Uses the binary malignant flag first; falls back to disease-name parsing.
    """
    if malignant_flag is True or malignant_flag == "True":
        return "malignant"
    # benign if disease maps to a benign HAM10000 class
    h = DDI_DISEASE_TO_HAM10K.get(str(disease).lower().strip())
    if h is not None and h in HAM10K_BENIGN:
        return "benign"
    # otherwise non-neoplastic (most of the 78 DDI diseases)
    return "non_neoplastic"


def ddi_to_ham10k(malignant_flag: bool, disease: str) -> str | None:
    """
    Map DDI sample to HAM10000 7-class. Returns None if no clear mapping
    (most DDI samples are inflammatory conditions outside HAM10000's scope).
    """
    return DDI_DISEASE_TO_HAM10K.get(str(disease).lower().strip())


# ── Fitzpatrick17k mapping ───────────────────────────────────────────────────
# Fitz17k provides 3 levels of granularity:
#   - label: 114 specific conditions
#   - nine_partition_label: 9 groups
#   - three_partition_label: 3 groups (benign-epidermal, malignant, etc.)
#
# We use three_partition_label for the universal mapping. The native 114-class
# label is used for diseases that overlap with HAM10000.

# Map Fitzpatrick17k's three_partition_label -> universal taxonomy
FITZ17K_PARTITION_TO_UNIVERSAL = {
    "benign":          "benign",
    "malignant":       "malignant",
    "non-neoplastic":  "non_neoplastic",
}

# Diseases in Fitzpatrick17k that map clearly to HAM10000 7-class.
# Anything not in this dict has no HAM10000 mapping.
FITZ17K_LABEL_TO_HAM10K = {
    # Melanoma
    "melanoma":                          "mel",
    "melanoma acral lentiginous":        "mel",
    # Nevi
    "nevus":                             "nv",
    "dysplastic nevus":                  "nv",
    # BCC
    "basal cell carcinoma":              "bcc",
    "basal cell carcinoma morpheiform":  "bcc",
    # SCC / actinic keratosis -> akiec
    "actinic keratosis":                 "akiec",
    "squamous cell carcinoma":           "akiec",
    # Seborrheic keratosis -> bkl
    "seborrheic keratosis":              "bkl",
    "solar lentigo":                     "bkl",
    # Dermatofibroma
    "dermatofibroma":                    "df",
    # Vascular
    "pyogenic granuloma":                "vasc",
    "hemangioma":                        "vasc",
    "kaposi sarcoma":                    "vasc",
}


def fitz17k_to_universal(three_partition: str) -> str:
    """Map Fitz17k three_partition_label to universal taxonomy."""
    return FITZ17K_PARTITION_TO_UNIVERSAL.get(
        str(three_partition).strip().lower(),
        "non_neoplastic",
    )


def fitz17k_to_ham10k(label: str) -> str | None:
    """Map Fitz17k specific label to HAM10000 7-class. None if no overlap."""
    return FITZ17K_LABEL_TO_HAM10K.get(str(label).strip().lower())


# ── Fitzpatrick skin tone bucketing ──────────────────────────────────────────
# We bucket the 1-6 Fitzpatrick scale into 3 levels for stratified eval.
# This is the standard practice in dermatology fairness papers.

FITZ_BUCKETS = {1: "light", 2: "light", 3: "medium", 4: "medium", 5: "dark", 6: "dark"}
FITZ_BUCKET_NAMES = ["light", "medium", "dark"]


def bucket_fitzpatrick(value) -> str | None:
    """Map raw Fitzpatrick 1-6 value to {'light','medium','dark'} or None."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return FITZ_BUCKETS.get(v)


def bucket_ddi_skintone(value: str) -> str | None:
    """
    DDI provides skin tone as a string like '12', '34', '56'.
    Map to {'light','medium','dark'}.
    """
    s = str(value).strip()
    if s in {"12", "1-2", "1,2"}:
        return "light"
    if s in {"34", "3-4", "3,4"}:
        return "medium"
    if s in {"56", "5-6", "5,6"}:
        return "dark"
    return None