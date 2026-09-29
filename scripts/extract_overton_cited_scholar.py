from pathlib import Path
import re

import pandas as pd


# =========================================================
# Project directories
# =========================================================

# project_root/
# ├── data/
# │   ├── overton/
# │   │   └── overton_raw.xlsx
# │   ├── scopus/
# │   │   └── scopus_full.xlsx
# │   └── lda/
# │       └── overton_from_scopus_scholar.xlsx
# │
# └── scripts/
#     └── extract_overton_cited_scholar.py


PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


OVERTON_DIR = (
    PROJECT_DIR
    / "data"
    / "overton"
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

OVERTON_FILE = (
    OVERTON_DIR
    / "overton_raw.xlsx"
)


SCOPUS_FILE = (
    SCOPUS_DIR
    / "scopus_full.xlsx"
)


OUTPUT_FILE = (
    LDA_DIR
    / "overton_from_scopus_scholar.xlsx"
)


# =========================================================
# Overton columns
# =========================================================

OVERTON_REQUIRED_COLUMNS = [
    "Title",
    "DOI",
    "Journal",
    "Published on",
    "Policy citation count",
    "Type",
    "Publisher",
    "Authors",
    "Your tags",
    "ORCIDs",
]


# =========================================================
# Scopus columns to append
# =========================================================

SCOPUS_REQUIRED_COLUMNS = [
    "DOI",
    "Authors",
    "Author full names",
    "Year",
    "Cited by",
    "Affiliations",
    "Abstract",
    "Author Keywords",
    "Index Keywords",
    "Funding Details",
    "Funding Texts",
]


# =========================================================
# Final output columns
# =========================================================

# "Overton Authors" is used for the Overton author field because
# Scopus also contains a column named "Authors".
#
# This avoids duplicate Excel column names.

FINAL_OUTPUT_COLUMNS = [
    "Title",
    "DOI",
    "Journal",
    "Published on",
    "Policy citation count",
    "Type",
    "Publisher",
    "Overton Authors",
    "Your tags",
    "ORCIDs",
    "Abstract",
    "Authors",
    "Author full names",
    "Year",
    "Cited by",
    "Affiliations",
    "Author Keywords",
    "Index Keywords",
    "Funding Details",
    "Funding Texts",
]


# =========================================================
# DOI normalization
# =========================================================

def normalize_doi(value):
    """
    Normalize a DOI for reliable matching.

    Examples
    --------
    DOI:10.1016/j.energy.2020.123456
        -> 10.1016/j.energy.2020.123456

    https://doi.org/10.1016/j.energy.2020.123456
        -> 10.1016/j.energy.2020.123456
    """

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
    # Remove DOI URL prefix
    # -----------------------------------------------------

    doi = re.sub(
        r"(?i)^https?://(?:dx\.)?doi\.org/",
        "",
        doi,
    )


    # -----------------------------------------------------
    # Normalize case and trailing punctuation
    # -----------------------------------------------------

    doi = (
        doi
        .strip()
        .rstrip(
            ".,;"
        )
        .lower()
    )


    return doi or None


# =========================================================
# Validate input files
# =========================================================

required_files = [
    OVERTON_FILE,
    SCOPUS_FILE,
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
    "Overton file      :",
    OVERTON_FILE
)

print(
    "Scopus file       :",
    SCOPUS_FILE
)

print(
    "Output file       :",
    OUTPUT_FILE
)

print()


# =========================================================
# Read Overton dataset
# =========================================================

overton = pd.read_excel(
    OVERTON_FILE
)


print("OVERTON DATASET")
print("=" * 70)

print(
    "Documents :",
    f"{len(overton):,}"
)

print(
    "Columns   :",
    f"{len(overton.columns):,}"
)

print()


# =========================================================
# Validate Overton columns
# =========================================================

missing_overton_columns = [
    column
    for column in OVERTON_REQUIRED_COLUMNS
    if column not in overton.columns
]


if missing_overton_columns:

    print(
        "Available Overton columns:"
    )

    for column in overton.columns:

        print(
            " -",
            column
        )


    raise ValueError(
        "Missing required Overton columns: "
        f"{missing_overton_columns}"
    )


# =========================================================
# Prepare Overton data
# =========================================================

overton = (
    overton
    .copy()
)


# Rename Overton Authors so it does not conflict with
# the Scopus Authors column.

overton = (
    overton
    .rename(
        columns={
            "Authors":
                "Overton Authors",
        }
    )
)


overton[
    "_doi_normalized"
] = (
    overton[
        "DOI"
    ]
    .apply(
        normalize_doi
    )
)


# =========================================================
# Overton DOI diagnostics
# =========================================================

n_overton_with_doi = (
    overton[
        "_doi_normalized"
    ]
    .notna()
    .sum()
)


n_overton_without_doi = (
    overton[
        "_doi_normalized"
    ]
    .isna()
    .sum()
)


n_overton_unique_dois = (
    overton[
        "_doi_normalized"
    ]
    .dropna()
    .nunique()
)


print("OVERTON DOI DIAGNOSTICS")
print("=" * 70)

print(
    "Rows with DOI    :",
    f"{n_overton_with_doi:,}"
)

print(
    "Rows without DOI :",
    f"{n_overton_without_doi:,}"
)

print(
    "Unique DOIs      :",
    f"{n_overton_unique_dois:,}"
)

print()


# =========================================================
# Show Overton rows without DOI
# =========================================================

if n_overton_without_doi > 0:

    print(
        "OVERTON ROWS WITHOUT DOI"
    )

    print(
        "=" * 70
    )


    print(
        overton.loc[
            overton[
                "_doi_normalized"
            ].isna(),
            [
                "Title",
                "DOI",
            ],
        ]
        .to_string(
            index=False
        )
    )

    print()


# =========================================================
# Check duplicate Overton DOI rows
# =========================================================

overton_duplicate_mask = (
    overton[
        "_doi_normalized"
    ]
    .notna()
    &
    overton[
        "_doi_normalized"
    ]
    .duplicated(
        keep=False
    )
)


n_overton_duplicate_rows = int(
    overton_duplicate_mask.sum()
)


print("OVERTON DUPLICATE DOI CHECK")
print("=" * 70)

print(
    "Duplicate DOI rows:",
    f"{n_overton_duplicate_rows:,}"
)

print()


if n_overton_duplicate_rows > 0:

    print(
        overton.loc[
            overton_duplicate_mask,
            [
                "Title",
                "DOI",
                "_doi_normalized",
                "Policy citation count",
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
    for column in SCOPUS_REQUIRED_COLUMNS
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
# Select only Scopus columns needed for the merge
# =========================================================

scopus_metadata = (
    scopus[
        [
            "_doi_normalized",
            "Authors",
            "Author full names",
            "Year",
            "Cited by",
            "Affiliations",
            "Abstract",
            "Author Keywords",
            "Index Keywords",
            "Funding Details",
            "Funding Texts",
        ]
    ]
    .copy()
)


# =========================================================
# Check duplicate Scopus DOIs
# =========================================================

scopus_duplicate_mask = (
    scopus_metadata[
        "_doi_normalized"
    ]
    .notna()
    &
    scopus_metadata[
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
    "Duplicate DOI rows:",
    f"{n_scopus_duplicate_rows:,}"
)

print()


if n_scopus_duplicate_rows > 0:

    duplicate_scopus_dois = (
        scopus.loc[
            scopus[
                "_doi_normalized"
            ]
            .notna()
            &
            scopus[
                "_doi_normalized"
            ]
            .duplicated(
                keep=False
            ),
            [
                "Title",
                "DOI",
                "_doi_normalized",
            ],
        ]
        .sort_values(
            "_doi_normalized"
        )
    )


    print(
        duplicate_scopus_dois
        .to_string(
            index=False
        )
    )

    print()


# =========================================================
# Keep one Scopus record per DOI
# =========================================================

scopus_metadata = (
    scopus_metadata
    .dropna(
        subset=[
            "_doi_normalized"
        ]
    )
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
# Match Overton DOI against Scopus
# =========================================================

overton_doi_set = set(
    overton[
        "_doi_normalized"
    ]
    .dropna()
)


scopus_doi_set = set(
    scopus_metadata[
        "_doi_normalized"
    ]
)


matched_dois = (
    overton_doi_set
    &
    scopus_doi_set
)


unmatched_dois = (
    overton_doi_set
    -
    scopus_doi_set
)


print("OVERTON-SCOPUS DOI MATCHING")
print("=" * 70)

print(
    "Overton unique DOIs :",
    f"{len(overton_doi_set):,}"
)

print(
    "Matched DOIs        :",
    f"{len(matched_dois):,}"
)

print(
    "Unmatched DOIs      :",
    f"{len(unmatched_dois):,}"
)


if len(overton_doi_set) > 0:

    match_rate = (
        len(matched_dois)
        /
        len(overton_doi_set)
        *
        100
    )

else:

    match_rate = 0.0


print(
    "DOI match rate      :",
    f"{match_rate:.2f}%"
)

print()


# =========================================================
# Select Overton publications recovered from Scopus
# =========================================================

matched_overton = (
    overton[
        overton[
            "_doi_normalized"
        ]
        .isin(
            matched_dois
        )
    ]
    .copy()
)


# =========================================================
# Merge Scopus scholarly metadata
# =========================================================

merged = (
    matched_overton
    .merge(
        scopus_metadata,
        on="_doi_normalized",
        how="left",
        validate="many_to_one",
    )
)


# =========================================================
# Keep requested output columns
# =========================================================

merged = (
    merged[
        FINAL_OUTPUT_COLUMNS
    ]
    .copy()
)


# =========================================================
# Sort output
# =========================================================

# Highest policy-cited scholarly publications first.
#
# Secondary sorting uses publication year and title only
# for deterministic output.

merged[
    "Policy citation count"
] = pd.to_numeric(
    merged[
        "Policy citation count"
    ],
    errors="coerce",
)


merged = (
    merged
    .sort_values(
        [
            "Policy citation count",
            "Year",
            "Title",
        ],
        ascending=[
            False,
            True,
            True,
        ],
        na_position="last",
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# Save output
# =========================================================

merged.to_excel(
    OUTPUT_FILE,
    index=False,
)


# =========================================================
# Final corpus summary
# =========================================================

print("FINAL OVERTON-SCOPUS SCHOLARLY CORPUS")
print("=" * 70)

print(
    "Documents        :",
    f"{len(merged):,}"
)

print(
    "Columns          :",
    f"{len(merged.columns):,}"
)

print(
    "Unique DOI       :",
    f"{merged['DOI'].apply(normalize_doi).nunique():,}"
)

print(
    "With abstract    :",
    f"{merged['Abstract'].notna().sum():,}"
)

print(
    "Missing abstract :",
    f"{merged['Abstract'].isna().sum():,}"
)

print()


# =========================================================
# Policy citation statistics
# =========================================================

print("POLICY CITATION STATISTICS")
print("=" * 70)


policy_counts = (
    merged[
        "Policy citation count"
    ]
)


print(
    "Total policy citations :",
    f"{policy_counts.sum():,.0f}"
)

print(
    "Mean per publication   :",
    f"{policy_counts.mean():.2f}"
)

print(
    "Median per publication :",
    f"{policy_counts.median():.2f}"
)

print(
    "Maximum                :",
    f"{policy_counts.max():,.0f}"
)

print()


# =========================================================
# Scopus citation statistics
# =========================================================

scopus_citations = pd.to_numeric(
    merged[
        "Cited by"
    ],
    errors="coerce",
)


print("SCOPUS CITATION STATISTICS")
print("=" * 70)

print(
    "Total Scopus citations :",
    f"{scopus_citations.sum():,.0f}"
)

print(
    "Mean per publication   :",
    f"{scopus_citations.mean():.2f}"
)

print(
    "Median per publication :",
    f"{scopus_citations.median():.2f}"
)

print(
    "Maximum                :",
    f"{scopus_citations.max():,.0f}"
)

print()


# =========================================================
# Top policy-cited publications
# =========================================================

print("TOP 20 POLICY-CITED PUBLICATIONS")
print("=" * 70)


top_policy_cited = (
    merged[
        [
            "Title",
            "DOI",
            "Journal",
            "Published on",
            "Policy citation count",
            "Publisher",
            "Year",
            "Cited by",
        ]
    ]
    .head(
        20
    )
    .copy()
)


top_policy_cited.insert(
    0,
    "Rank",
    range(
        1,
        len(top_policy_cited) + 1,
    ),
)


print(
    top_policy_cited
    .to_string(
        index=False
    )
)

print()


# =========================================================
# Unmatched DOI diagnostics
# =========================================================

if unmatched_dois:

    print("OVERTON DOIs NOT FOUND IN SCOPUS")
    print("=" * 70)


    unmatched_overton = (
        overton[
            overton[
                "_doi_normalized"
            ]
            .isin(
                unmatched_dois
            )
        ][
            [
                "Title",
                "DOI",
                "Journal",
                "Published on",
                "Policy citation count",
            ]
        ]
        .sort_values(
            [
                "Policy citation count",
                "Title",
            ],
            ascending=[
                False,
                True,
            ],
            na_position="last",
        )
    )


    print(
        unmatched_overton
        .to_string(
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
# Validation
# =========================================================

errors = []


if n_overton_without_doi > 0:

    errors.append(
        f"{n_overton_without_doi:,} Overton record(s) "
        "do not contain a DOI."
    )


if unmatched_dois:

    errors.append(
        f"{len(unmatched_dois):,} unique Overton DOI(s) "
        "were not recovered from Scopus."
    )


output_normalized_dois = (
    merged[
        "DOI"
    ]
    .apply(
        normalize_doi
    )
)


if output_normalized_dois.duplicated().any():

    errors.append(
        "Final output contains duplicate DOI values."
    )


if len(merged) != len(
    output_normalized_dois.dropna().unique()
):

    errors.append(
        "Final output does not contain exactly "
        "one row per matched DOI."
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


else:

    print("CHECK PASSED")
    print("=" * 70)

    print(
        "All Overton records contain a DOI."
    )

    print(
        "All unique Overton DOIs were recovered "
        "from Scopus."
    )

    print(
        "The final output contains one unique "
        "scholarly publication per DOI."
    )