from pathlib import Path
import re

import pandas as pd


# =========================================================
# Configuration
# =========================================================

# Script location:
#
# project_root/
# ├── data/
# │   └── lens/
# │       └── lenspatcite-export-....xlsx
# └── scripts/
#     └── extract_lens_dois.py
#
# Therefore the project root is one level above scripts/.

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

LENS_DIR = (
    PROJECT_DIR
    / "data"
    / "lens"
)

INPUT_FILE = (
    LENS_DIR
    / "lenspatcite-export-M_Muna-84f6bb81-32bc-4ec9-97c6-f970293a9752-2026-09-25_01-48-07_cited.xlsx"
)

OUTPUT_FILE = (
    LENS_DIR
    / "lens_patent_cited_dois.txt"
)

# Lens column containing DOI + other identifiers.
EXTERNAL_ID_COLUMN = (
    "citation external id"
)


# =========================================================
# Validate input
# =========================================================

print("Project directory :", PROJECT_DIR)
print("Lens directory    :", LENS_DIR)
print("Input file        :", INPUT_FILE)
print()

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found: "
        f"{INPUT_FILE}"
    )


# =========================================================
# Load Lens export
# =========================================================

df = pd.read_excel(
    INPUT_FILE
)

print("Lens export")
print("-----------")

print(
    f"Rows    : "
    f"{len(df):,}"
)

print(
    f"Columns : "
    f"{len(df.columns):,}"
)

print()


# =========================================================
# Validate required Lens column
# =========================================================

if EXTERNAL_ID_COLUMN not in df.columns:

    print(
        "Available columns:"
    )

    for column in df.columns:
        print(
            f"  - {column}"
        )

    raise ValueError(
        f"Required column "
        f"'{EXTERNAL_ID_COLUMN}' "
        f"was not found."
    )

print(
    "External ID column:",
    EXTERNAL_ID_COLUMN
)

print()


# =========================================================
# DOI extraction
# =========================================================

# Example Lens value:
#
# DOI:10.1016/j.neucom.2013.03.029
# MAGID:magid:1985230864
#
# Extract only the DOI appearing after "DOI:".
#
# The DOI stops at whitespace/newline or another identifier.

DOI_PATTERN = re.compile(
    r"(?i)\bDOI:\s*"
    r"(10\.\d{4,9}/[^\s]+)"
)


def extract_doi(value):
    """
    Extract a DOI from Lens 'Citation External Id'.

    Returns
    -------
    str or None
        Normalized lowercase DOI.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    match = DOI_PATTERN.search(
        text
    )

    if match is None:
        return None

    doi = (
        match.group(1)
        .strip()
        .rstrip(
            ".,;"
        )
        .lower()
    )

    return doi


df["clean_doi"] = (
    df[EXTERNAL_ID_COLUMN]
    .apply(
        extract_doi
    )
)


# =========================================================
# DOI diagnostics
# =========================================================

n_documents = len(df)

n_with_doi = (
    df["clean_doi"]
    .notna()
    .sum()
)

n_without_doi = (
    df["clean_doi"]
    .isna()
    .sum()
)

n_unique_dois = (
    df["clean_doi"]
    .dropna()
    .nunique()
)

n_duplicate_rows = (
    df.loc[
        df["clean_doi"].notna(),
        "clean_doi",
    ]
    .duplicated(
        keep=False
    )
    .sum()
)

n_duplicate_extra = (
    n_with_doi
    - n_unique_dois
)


print("DOI extraction")
print("--------------")

print(
    f"Scholar documents    : "
    f"{n_documents:,}"
)

print(
    f"Rows with DOI        : "
    f"{n_with_doi:,}"
)

print(
    f"Rows without DOI     : "
    f"{n_without_doi:,}"
)

print(
    f"Unique DOIs          : "
    f"{n_unique_dois:,}"
)

print(
    f"Duplicate DOI rows   : "
    f"{n_duplicate_rows:,}"
)

print(
    f"Duplicate DOI excess : "
    f"{n_duplicate_extra:,}"
)

print()


# =========================================================
# Inspect rows without DOI
# =========================================================

if n_without_doi > 0:

    print(
        "Rows without a DOI"
    )

    print(
        "------------------"
    )

    # Try to include a title-like field if Lens provides one.
    title_candidates = [
        "Citation Title",
        "Title",
    ]

    columns_to_show = [
        column
        for column in title_candidates
        if column in df.columns
    ]

    columns_to_show.append(
        EXTERNAL_ID_COLUMN
    )

    print(
        df.loc[
            df["clean_doi"].isna(),
            columns_to_show,
        ]
        .to_string(
            index=False
        )
    )

    print()


# =========================================================
# Inspect duplicated DOIs
# =========================================================

duplicate_mask = (
    df["clean_doi"]
    .notna()
    &
    df["clean_doi"]
    .duplicated(
        keep=False
    )
)

if duplicate_mask.any():

    print(
        "Duplicated DOIs"
    )

    print(
        "---------------"
    )

    title_candidates = [
        "Citation Title",
        "Title",
    ]

    columns_to_show = [
        column
        for column in title_candidates
        if column in df.columns
    ]

    columns_to_show.append(
        "clean_doi"
    )

    print(
        df.loc[
            duplicate_mask,
            columns_to_show,
        ]
        .sort_values(
            "clean_doi"
        )
        .to_string(
            index=False
        )
    )

    print()


# =========================================================
# Save one UNIQUE DOI per line
# =========================================================

dois = (
    df["clean_doi"]
    .dropna()
    .drop_duplicates()
    .tolist()
)

OUTPUT_FILE.write_text(
    "\n".join(dois)
    + (
        "\n"
        if dois
        else ""
    ),
    encoding="utf-8",
)


# =========================================================
# Final summary
# =========================================================

print("Output")
print("------")

print(
    "Saved:",
    OUTPUT_FILE
)

print(
    f"DOIs written: "
    f"{len(dois):,}"
)

print()


# =========================================================
# One-document / one-DOI validation
# =========================================================

if (
    n_with_doi == n_documents
    and
    n_unique_dois == n_documents
):

    print(
        "CHECK PASSED"
    )

    print(
        "------------"
    )

    print(
        "Every scholarly document has "
        "exactly one unique DOI."
    )

else:

    print(
        "CHECK WARNING"
    )

    print(
        "-------------"
    )

    print(
        f"Lens contains "
        f"{n_documents:,} scholarly documents, "
        f"but only {n_unique_dois:,} unique DOIs "
        f"were extracted."
    )

    if n_without_doi > 0:

        print(
            f"- {n_without_doi:,} document(s) "
            f"do not contain an extractable DOI."
        )

    if n_duplicate_extra > 0:

        print(
            f"- {n_duplicate_extra:,} DOI occurrence(s) "
            f"are duplicates."
        )