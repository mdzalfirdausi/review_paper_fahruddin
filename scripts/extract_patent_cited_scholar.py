from pathlib import Path
import re

import pandas as pd


# =========================================================
# Project directories
# =========================================================

# project_root/
# ├── data/
# │   ├── lens/
# │   │   ├── lens_patent_cited_dois.txt
# │   │   └── lenspatcite-export-....xlsx
# │   ├── scopus/
# │   │   └── scopus_full.xlsx
# │   └── lda/
# │       └── lens_from_scopus_scholar.xlsx
# │
# └── scripts/
#     └── extract_patent_cited_scholar.py


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


SCOPUS_DIR = (
    PROJECT_DIR
    / "data"
    / "scopus"
)


LDA_DIR = (
    PROJECT_DIR
    / "data"
    / "lda"
)


LDA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Files
# =========================================================

DOI_FILE = (
    LENS_DIR
    / "lens_patent_cited_dois.txt"
)


SCOPUS_FILE = (
    SCOPUS_DIR
    / "scopus_full.xlsx"
)


LENS_EXPORT_FILE = (
    LENS_DIR
    / (
        "lenspatcite-export-M_Muna-84f6bb81-32bc-4ec9-97c6-f970293a9752-2026-09-25_01-48-07_cited.xlsx"
    )
)


OUTPUT_FILE = (
    LDA_DIR
    / "lens_from_scopus_scholar.xlsx"
)


# =========================================================
# Scopus output columns
# =========================================================

SCOPUS_OUTPUT_COLUMNS = [
    "Link",
    "Authors",
    "Author full names",
    "Author(s) ID",
    "Title",
    "Year",
    "Source title",
    "Cited by",
    "Affiliations",
    "Publisher",
    "Abbreviated Source Title",
    "DOI",
    "Abstract",
    "Author Keywords",
    "Index Keywords",
    "Funding Details",
    "Funding Texts",
]


# =========================================================
# Lens columns
# =========================================================

LENS_EXTERNAL_ID_COLUMN = (
    "citation external id"
)


LENS_PATENT_COUNT_COLUMN = (
    "cited by patent count"
)


LENS_FAMILY_COUNT_COLUMN = (
    "citing family count"
)


LENS_REQUIRED_COLUMNS = [
    LENS_EXTERNAL_ID_COLUMN,
    LENS_PATENT_COUNT_COLUMN,
    LENS_FAMILY_COUNT_COLUMN,
]


# =========================================================
# Final Lens-derived output columns
# =========================================================

LENS_OUTPUT_COLUMNS = [
    "Patent citation count",
    "Citing patent family count",
]


# =========================================================
# DOI normalization
# =========================================================

def normalize_doi(value):

    if pd.isna(value):

        return None


    doi = str(
        value
    ).strip()


    if not doi:

        return None


    # -----------------------------------------------------
    # Remove DOI: prefix
    # -----------------------------------------------------

    doi = re.sub(
        r"(?i)^doi:\s*",
        "",
        doi,
    )


    # -----------------------------------------------------
    # Remove DOI URL
    # -----------------------------------------------------

    doi = re.sub(
        r"(?i)^https?://(?:dx\.)?doi\.org/",
        "",
        doi,
    )


    doi = (
        doi
        .strip()
        .rstrip(".,;")
        .lower()
    )


    return doi or None


# =========================================================
# Extract DOI from Lens citation external ID
#
# Example:
#
# DOI:10.1016/j.neucom.2013.03.029
# MAGID:magid:1985230864
# =========================================================

DOI_PATTERN = re.compile(
    r"(?i)\bDOI:\s*"
    r"(10\.\d{4,9}/[^\s]+)"
)


def extract_lens_doi(value):

    if pd.isna(value):

        return None


    text = str(
        value
    ).strip()


    match = DOI_PATTERN.search(
        text
    )


    if match is None:

        return None


    doi = (
        match
        .group(1)
        .strip()
        .rstrip(".,;")
        .lower()
    )


    return doi or None


# =========================================================
# Validate input files
# =========================================================

required_files = [
    DOI_FILE,
    SCOPUS_FILE,
    LENS_EXPORT_FILE,
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"File not found:\n"
            f"{file_path}"
        )


print("PROJECT FILES")
print("=" * 70)

print(
    "Project directory :",
    PROJECT_DIR
)

print(
    "DOI file          :",
    DOI_FILE
)

print(
    "Scopus file       :",
    SCOPUS_FILE
)

print(
    "Lens export       :",
    LENS_EXPORT_FILE
)

print(
    "Output file       :",
    OUTPUT_FILE
)

print()


# =========================================================
# Read Lens patent-cited DOI list
# =========================================================

raw_dois = (
    DOI_FILE
    .read_text(
        encoding="utf-8"
    )
    .splitlines()
)


lens_dois = [
    normalize_doi(
        value
    )
    for value in raw_dois
]


lens_dois = [
    doi
    for doi in lens_dois
    if doi is not None
]


lens_doi_set = set(
    lens_dois
)


print("LENS PATENT-CITED DOI LIST")
print("=" * 70)

print(
    "DOI lines   :",
    f"{len(raw_dois):,}"
)

print(
    "Valid DOIs  :",
    f"{len(lens_dois):,}"
)

print(
    "Unique DOIs :",
    f"{len(lens_doi_set):,}"
)

print()


# =========================================================
# Read full Scopus dataset
# =========================================================

scopus = pd.read_excel(
    SCOPUS_FILE
)


print("SCOPUS FULL DATASET")
print("=" * 70)

print(
    "Documents :",
    f"{len(scopus):,}"
)

print(
    "Columns   :",
    f"{len(scopus.columns):,}"
)

print()


# =========================================================
# Validate Scopus columns
# =========================================================

missing_scopus_columns = [
    column
    for column in SCOPUS_OUTPUT_COLUMNS
    if column not in scopus.columns
]


if missing_scopus_columns:

    print(
        "Available Scopus columns:"
    )

    for column in scopus.columns:

        print(
            " -",
            column
        )


    raise ValueError(
        "Missing required Scopus columns: "
        f"{missing_scopus_columns}"
    )


# =========================================================
# Normalize Scopus DOI
# =========================================================

scopus = (
    scopus
    .copy()
)


scopus[
    "_doi_normalized"
] = (
    scopus[
        "DOI"
    ]
    .apply(
        normalize_doi
    )
)


# =========================================================
# Read raw Lens PatCite export
# =========================================================

lens = pd.read_excel(
    LENS_EXPORT_FILE
)


print("LENS PATCITE EXPORT")
print("=" * 70)

print(
    "Rows    :",
    f"{len(lens):,}"
)

print(
    "Columns :",
    f"{len(lens.columns):,}"
)

print()


# =========================================================
# Validate Lens columns
# =========================================================

missing_lens_columns = [
    column
    for column in LENS_REQUIRED_COLUMNS
    if column not in lens.columns
]


if missing_lens_columns:

    print(
        "Available Lens columns:"
    )

    for column in lens.columns:

        print(
            " -",
            column
        )


    raise ValueError(
        "Missing required Lens columns: "
        f"{missing_lens_columns}"
    )


# =========================================================
# Extract Lens DOI
# =========================================================

lens = (
    lens
    .copy()
)


lens[
    "_doi_normalized"
] = (
    lens[
        LENS_EXTERNAL_ID_COLUMN
    ]
    .apply(
        extract_lens_doi
    )
)


# =========================================================
# Lens DOI diagnostics
# =========================================================

lens_rows_with_doi = (
    lens[
        "_doi_normalized"
    ]
    .notna()
    .sum()
)


lens_unique_dois = (
    lens[
        "_doi_normalized"
    ]
    .dropna()
    .nunique()
)


lens_missing_dois = (
    lens[
        "_doi_normalized"
    ]
    .isna()
    .sum()
)


print("LENS DOI EXTRACTION")
print("=" * 70)

print(
    "Rows with DOI    :",
    f"{lens_rows_with_doi:,}"
)

print(
    "Unique DOIs      :",
    f"{lens_unique_dois:,}"
)

print(
    "Rows without DOI :",
    f"{lens_missing_dois:,}"
)

print()


# =========================================================
# Verify Lens export and DOI list agree
# =========================================================

lens_export_doi_set = set(
    lens[
        "_doi_normalized"
    ]
    .dropna()
)


doi_list_only = (
    lens_doi_set
    -
    lens_export_doi_set
)


lens_export_only = (
    lens_export_doi_set
    -
    lens_doi_set
)


print("LENS DOI CONSISTENCY")
print("=" * 70)

print(
    "DOI-list unique DOIs  :",
    f"{len(lens_doi_set):,}"
)

print(
    "Lens-export unique DOIs:",
    f"{len(lens_export_doi_set):,}"
)

print(
    "Only in DOI list       :",
    f"{len(doi_list_only):,}"
)

print(
    "Only in Lens export    :",
    f"{len(lens_export_only):,}"
)

print()


# =========================================================
# Convert Lens patent metrics to numeric
# =========================================================

lens[
    LENS_PATENT_COUNT_COLUMN
] = pd.to_numeric(
    lens[
        LENS_PATENT_COUNT_COLUMN
    ],
    errors="coerce",
)


lens[
    LENS_FAMILY_COUNT_COLUMN
] = pd.to_numeric(
    lens[
        LENS_FAMILY_COUNT_COLUMN
    ],
    errors="coerce",
)


# =========================================================
# Check duplicate Lens DOIs
# =========================================================

lens_duplicate_mask = (
    lens[
        "_doi_normalized"
    ]
    .notna()
    &
    lens[
        "_doi_normalized"
    ]
    .duplicated(
        keep=False
    )
)


n_lens_duplicate_rows = int(
    lens_duplicate_mask.sum()
)


print("LENS DUPLICATE DOI CHECK")
print("=" * 70)

print(
    "Duplicate DOI rows:",
    f"{n_lens_duplicate_rows:,}"
)

print()


if n_lens_duplicate_rows > 0:

    duplicate_lens = (
        lens.loc[
            lens_duplicate_mask,
            [
                "_doi_normalized",
                LENS_PATENT_COUNT_COLUMN,
                LENS_FAMILY_COUNT_COLUMN,
            ],
        ]
        .sort_values(
            "_doi_normalized"
        )
    )


    print(
        duplicate_lens.to_string(
            index=False
        )
    )

    print()


# =========================================================
# Prepare Lens patent-impact metrics
#
# If duplicate DOI rows unexpectedly occur, aggregate rather
# than silently selecting one row.
#
# For identical scholarly works represented multiple times,
# max() avoids double-counting an already aggregated Lens
# patent-citation count.
# =========================================================

lens_metrics = (
    lens[
        [
            "_doi_normalized",
            LENS_PATENT_COUNT_COLUMN,
            LENS_FAMILY_COUNT_COLUMN,
        ]
    ]
    .dropna(
        subset=[
            "_doi_normalized"
        ]
    )
    .groupby(
        "_doi_normalized",
        as_index=False,
    )
    .agg(
        {
            LENS_PATENT_COUNT_COLUMN:
                "max",

            LENS_FAMILY_COUNT_COLUMN:
                "max",
        }
    )
    .rename(
        columns={
            LENS_PATENT_COUNT_COLUMN:
                "Patent citation count",

            LENS_FAMILY_COUNT_COLUMN:
                "Citing patent family count",
        }
    )
)


print("LENS PATENT METRICS")
print("=" * 70)

print(
    "Metric rows:",
    f"{len(lens_metrics):,}"
)

print(
    "Missing patent counts:",
    f"{lens_metrics['Patent citation count'].isna().sum():,}"
)

print(
    "Missing family counts:",
    f"{lens_metrics['Citing patent family count'].isna().sum():,}"
)

print()


# =========================================================
# Select Lens patent-cited publications from Scopus
# =========================================================

patent_cited = (
    scopus[
        scopus[
            "_doi_normalized"
        ]
        .isin(
            lens_doi_set
        )
    ]
    .copy()
)


# =========================================================
# Check Scopus matching
# =========================================================

matched_dois = set(
    patent_cited[
        "_doi_normalized"
    ]
    .dropna()
)


unmatched_dois = (
    lens_doi_set
    -
    matched_dois
)


print("SCOPUS DOI MATCHING")
print("=" * 70)

print(
    "Requested DOIs :",
    f"{len(lens_doi_set):,}"
)

print(
    "Matched DOIs   :",
    f"{len(matched_dois):,}"
)

print(
    "Matched rows   :",
    f"{len(patent_cited):,}"
)

print(
    "Unmatched DOIs :",
    f"{len(unmatched_dois):,}"
)

print()


# =========================================================
# Check duplicate Scopus DOI rows
# =========================================================

scopus_duplicate_mask = (
    patent_cited[
        "_doi_normalized"
    ]
    .duplicated(
        keep=False
    )
)


n_scopus_duplicate_rows = int(
    scopus_duplicate_mask.sum()
)


print("SCOPUS DUPLICATE DOI CHECK")
print("=" * 70)

print(
    "Duplicate matched rows:",
    f"{n_scopus_duplicate_rows:,}"
)

print()


if n_scopus_duplicate_rows > 0:

    print(
        patent_cited.loc[
            scopus_duplicate_mask,
            [
                "Title",
                "DOI",
                "_doi_normalized",
            ],
        ]
        .sort_values(
            "_doi_normalized"
        )
        .to_string(
            index=False
        )
    )

    print()


# =========================================================
# Keep one Scopus publication per DOI
# =========================================================

patent_cited = (
    patent_cited
    .drop_duplicates(
        subset=[
            "_doi_normalized"
        ],
        keep="first",
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# Merge Lens patent-impact metrics into Scopus metadata
# =========================================================

patent_cited = (
    patent_cited
    .merge(
        lens_metrics,
        on="_doi_normalized",
        how="left",
        validate="one_to_one",
    )
)


# =========================================================
# Validate merged patent metrics
# =========================================================

missing_patent_metric = (
    patent_cited[
        "Patent citation count"
    ]
    .isna()
)


missing_family_metric = (
    patent_cited[
        "Citing patent family count"
    ]
    .isna()
)


print("PATENT METRIC MERGE")
print("=" * 70)

print(
    "Publications:",
    f"{len(patent_cited):,}"
)

print(
    "With patent citation count:",
    f"{patent_cited['Patent citation count'].notna().sum():,}"
)

print(
    "Missing patent citation count:",
    f"{missing_patent_metric.sum():,}"
)

print(
    "With patent family count:",
    f"{patent_cited['Citing patent family count'].notna().sum():,}"
)

print(
    "Missing patent family count:",
    f"{missing_family_metric.sum():,}"
)

print()


if missing_patent_metric.any():

    print(
        "Publications missing patent citation metrics:"
    )

    print(
        patent_cited.loc[
            missing_patent_metric,
            [
                "Title",
                "DOI",
                "_doi_normalized",
            ],
        ]
        .to_string(
            index=False
        )
    )

    print()


# =========================================================
# Keep final requested columns
# =========================================================

FINAL_OUTPUT_COLUMNS = (
    SCOPUS_OUTPUT_COLUMNS
    +
    LENS_OUTPUT_COLUMNS
)


patent_cited = (
    patent_cited[
        FINAL_OUTPUT_COLUMNS
    ]
    .copy()
)


# =========================================================
# Sort output
#
# Highest patent-cited publications first.
# This does not affect subsequent DOI-based analyses.
# =========================================================

patent_cited = (
    patent_cited
    .sort_values(
        [
            "Patent citation count",
            "Citing patent family count",
            "Year",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        na_position="last",
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# Save Excel
# =========================================================

patent_cited.to_excel(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# Final corpus validation
# =========================================================

print("FINAL PATENT-CITED SCHOLARLY CORPUS")
print("=" * 70)

print(
    "Documents        :",
    f"{len(patent_cited):,}"
)

print(
    "Columns          :",
    f"{len(patent_cited.columns):,}"
)

print(
    "With abstract    :",
    f"{patent_cited['Abstract'].notna().sum():,}"
)

print(
    "Missing abstract :",
    f"{patent_cited['Abstract'].isna().sum():,}"
)

print(
    "Unique DOI       :",
    f"{patent_cited['DOI'].nunique():,}"
)

print(
    "Patent counts    :",
    f"{patent_cited['Patent citation count'].notna().sum():,}"
)

print(
    "Family counts    :",
    f"{patent_cited['Citing patent family count'].notna().sum():,}"
)

print()


# =========================================================
# Patent citation statistics
# =========================================================

print("PATENT CITATION STATISTICS")
print("=" * 70)

print(
    "Total patent citations :",
    f"{patent_cited['Patent citation count'].sum():,.0f}"
)

print(
    "Mean per publication   :",
    f"{patent_cited['Patent citation count'].mean():.2f}"
)

print(
    "Median per publication :",
    f"{patent_cited['Patent citation count'].median():.2f}"
)

print(
    "Maximum                :",
    f"{patent_cited['Patent citation count'].max():,.0f}"
)

print()

print(
    "Total citing families  :",
    f"{patent_cited['Citing patent family count'].sum():,.0f}"
)

print(
    "Mean families/article  :",
    f"{patent_cited['Citing patent family count'].mean():.2f}"
)

print(
    "Median families/article:",
    f"{patent_cited['Citing patent family count'].median():.2f}"
)

print(
    "Maximum families       :",
    f"{patent_cited['Citing patent family count'].max():,.0f}"
)

print()


# =========================================================
# Top patent-cited publications
# =========================================================

print("TOP 20 PATENT-CITED PUBLICATIONS")
print("=" * 70)


top_patent_cited = (
    patent_cited[
        [
            "Title",
            "Authors",
            "Year",
            "Source title",
            "DOI",
            "Patent citation count",
            "Citing patent family count",
            "Cited by",
        ]
    ]
    .head(
        20
    )
    .copy()
)


top_patent_cited.insert(
    0,
    "Rank",
    range(
        1,
        len(top_patent_cited) + 1,
    ),
)


print(
    top_patent_cited.to_string(
        index=False
    )
)

print()


# =========================================================
# Save confirmation
# =========================================================

print("OUTPUT")
print("=" * 70)

print(
    "Saved:",
    OUTPUT_FILE
)

print()


# =========================================================
# Strict expected-result checks
# =========================================================

errors = []


if len(matched_dois) != len(
    lens_doi_set
):

    errors.append(
        f"{len(unmatched_dois):,} Lens DOI(s) "
        "were not recovered from Scopus."
    )


if len(patent_cited) != len(
    lens_doi_set
):

    errors.append(
        "Final scholarly corpus does not contain "
        "one row per Lens DOI."
    )


if patent_cited[
    "DOI"
].nunique() != len(
    patent_cited
):

    errors.append(
        "Final output contains duplicate DOI values."
    )


if patent_cited[
    "Patent citation count"
].isna().any():

    errors.append(
        "Some publications are missing "
        "Patent citation count."
    )


if patent_cited[
    "Citing patent family count"
].isna().any():

    errors.append(
        "Some publications are missing "
        "Citing patent family count."
    )


# =========================================================
# Final status
# =========================================================

if errors:

    print("CHECK WARNING")
    print("=" * 70)

    for error in errors:

        print(
            " -",
            error
        )


    if unmatched_dois:

        print()
        print(
            "Unmatched Lens DOIs:"
        )

        for doi in sorted(
            unmatched_dois
        ):

            print(
                " -",
                doi
            )

else:

    print("CHECK PASSED")
    print("=" * 70)

    print(
        "All Lens patent-cited scholarly publications "
        "were recovered from Scopus."
    )

    print(
        "All publications were matched to Lens patent "
        "citation and patent-family metrics."
    )

    print(
        "The final output contains one unique scholarly "
        "publication per DOI."
    )