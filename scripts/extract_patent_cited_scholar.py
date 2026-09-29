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
# └── scripts/
#     └── extract_patent_cited_scholar.py

PROJECT_DIR = Path(__file__).resolve().parent.parent
LENS_DIR = PROJECT_DIR / "data" / "lens"
SCOPUS_DIR = PROJECT_DIR / "data" / "scopus"
LDA_DIR = PROJECT_DIR / "data" / "lda"
LDA_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Files
# =========================================================

DOI_FILE = LENS_DIR / "lens_patent_cited_dois.txt"
SCOPUS_FILE = SCOPUS_DIR / "scopus_full.xlsx"
LENS_EXPORT_FILE = LENS_DIR / (
    "lenspatcite-export-M_Muna-84f6bb81-32bc-4ec9-97c6-"
    "f970293a9752-2026-09-25_01-48-07_cited.xlsx"
)
OUTPUT_FILE = LDA_DIR / "lens_from_scopus_scholar.xlsx"


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

LENS_EXTERNAL_ID_COLUMN = "citation external id"
LENS_PATENT_COUNT_COLUMN = "cited by patent count"
LENS_FAMILY_COUNT_COLUMN = "citing family count"

LENS_REQUIRED_COLUMNS = [
    LENS_EXTERNAL_ID_COLUMN,
    LENS_PATENT_COUNT_COLUMN,
    LENS_FAMILY_COUNT_COLUMN,
]

LENS_OUTPUT_COLUMNS = [
    "Patent citation count",
    "Citing patent family count",
]


# =========================================================
# DOI normalization and extraction
# =========================================================

DOI_PATTERN = re.compile(
    r"(?i)\bDOI:\s*"
    r"(10\.\d{4,9}/[^\s]+)"
)


def normalize_doi(value):
    """Normalize a DOI value to a lowercase bare DOI string."""

    if pd.isna(value):
        return None

    doi = str(value).strip()

    if not doi:
        return None

    doi = re.sub(r"(?i)^doi:\s*", "", doi)
    doi = re.sub(r"(?i)^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = doi.strip().rstrip(".,;").lower()

    return doi or None


def extract_lens_dois(value):
    """
    Extract ALL DOI identifiers from one Lens 'citation external id' cell.

    Example
    -------
    Input cell:
        DOI:10.20944/preprints202507.1980.v1
        DOI:10.3390/s25185646
        PMID:41012884
        PMCID:pmc12473902

    Output:
        [
            '10.20944/preprints202507.1980.v1',
            '10.3390/s25185646',
        ]

    Repeated DOI values inside the same cell are removed while preserving
    their first-seen order.
    """

    if pd.isna(value):
        return []

    matches = DOI_PATTERN.findall(str(value).strip())
    dois = []
    seen = set()

    for match in matches:
        doi = normalize_doi(match)

        if doi is None or doi in seen:
            continue

        seen.add(doi)
        dois.append(doi)

    return dois


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
        raise FileNotFoundError(f"File not found:\n{file_path}")

print("PROJECT FILES")
print("=" * 70)
print("Project directory :", PROJECT_DIR)
print("DOI file          :", DOI_FILE)
print("Scopus file       :", SCOPUS_FILE)
print("Lens export       :", LENS_EXPORT_FILE)
print("Output file       :", OUTPUT_FILE)
print()


# =========================================================
# Read DOI list produced by extract_lens_dois.py
# =========================================================

raw_dois = DOI_FILE.read_text(encoding="utf-8").splitlines()

lens_dois = [normalize_doi(value) for value in raw_dois]
lens_dois = [doi for doi in lens_dois if doi is not None]
lens_doi_set = set(lens_dois)

print("LENS PATENT-CITED DOI LIST")
print("=" * 70)
print("DOI lines   :", f"{len(raw_dois):,}")
print("Valid DOIs  :", f"{len(lens_dois):,}")
print("Unique DOIs :", f"{len(lens_doi_set):,}")
print()


# =========================================================
# Read full Scopus dataset
# =========================================================

scopus = pd.read_excel(SCOPUS_FILE)

print("SCOPUS FULL DATASET")
print("=" * 70)
print("Documents :", f"{len(scopus):,}")
print("Columns   :", f"{len(scopus.columns):,}")
print()

missing_scopus_columns = [
    column
    for column in SCOPUS_OUTPUT_COLUMNS
    if column not in scopus.columns
]

if missing_scopus_columns:
    print("Available Scopus columns:")
    for column in scopus.columns:
        print(" -", column)

    raise ValueError(
        "Missing required Scopus columns: "
        f"{missing_scopus_columns}"
    )

scopus = scopus.copy()
scopus["_doi_normalized"] = scopus["DOI"].apply(normalize_doi)


# =========================================================
# Read raw Lens PatCite export
# =========================================================

lens = pd.read_excel(LENS_EXPORT_FILE)

print("LENS PATCITE EXPORT")
print("=" * 70)
print("Rows    :", f"{len(lens):,}")
print("Columns :", f"{len(lens.columns):,}")
print()

missing_lens_columns = [
    column
    for column in LENS_REQUIRED_COLUMNS
    if column not in lens.columns
]

if missing_lens_columns:
    print("Available Lens columns:")
    for column in lens.columns:
        print(" -", column)

    raise ValueError(
        "Missing required Lens columns: "
        f"{missing_lens_columns}"
    )


# =========================================================
# Extract ALL DOI aliases from every Lens scholarly record
# =========================================================

lens = lens.copy().reset_index(drop=True)
lens["_lens_record_id"] = lens.index
lens["_doi_list"] = lens[LENS_EXTERNAL_ID_COLUMN].apply(extract_lens_dois)
lens["_doi_count"] = lens["_doi_list"].apply(len)

lens_rows_with_doi = int(lens["_doi_count"].gt(0).sum())
lens_rows_without_doi = int(lens["_doi_count"].eq(0).sum())
lens_rows_multiple_dois = int(lens["_doi_count"].gt(1).sum())
lens_doi_occurrences = int(lens["_doi_count"].sum())

# Explode aliases so every DOI of a Lens record points back to the same
# Lens record and therefore inherits the same patent-impact metrics.
lens_aliases = (
    lens[
        [
            "_lens_record_id",
            "_doi_list",
            LENS_PATENT_COUNT_COLUMN,
            LENS_FAMILY_COUNT_COLUMN,
        ]
    ]
    .explode("_doi_list")
    .rename(columns={"_doi_list": "_doi_normalized"})
)

lens_aliases = lens_aliases.dropna(subset=["_doi_normalized"]).copy()

lens_export_doi_set = set(lens_aliases["_doi_normalized"])

print("LENS DOI EXTRACTION")
print("=" * 70)
print("Scholarly records      :", f"{len(lens):,}")
print("Records with DOI       :", f"{lens_rows_with_doi:,}")
print("Records without DOI    :", f"{lens_rows_without_doi:,}")
print("Records with >1 DOI    :", f"{lens_rows_multiple_dois:,}")
print("DOI occurrences        :", f"{lens_doi_occurrences:,}")
print("Unique DOI aliases     :", f"{len(lens_export_doi_set):,}")
print()


# =========================================================
# Verify DOI TXT and Lens export agree
# =========================================================

doi_list_only = lens_doi_set - lens_export_doi_set
lens_export_only = lens_export_doi_set - lens_doi_set

print("LENS DOI CONSISTENCY")
print("=" * 70)
print("DOI-list unique DOIs   :", f"{len(lens_doi_set):,}")
print("Lens-export DOI aliases:", f"{len(lens_export_doi_set):,}")
print("Only in DOI list       :", f"{len(doi_list_only):,}")
print("Only in Lens export    :", f"{len(lens_export_only):,}")
print()


# =========================================================
# Convert Lens patent metrics to numeric
# =========================================================

lens_aliases[LENS_PATENT_COUNT_COLUMN] = pd.to_numeric(
    lens_aliases[LENS_PATENT_COUNT_COLUMN],
    errors="coerce",
)

lens_aliases[LENS_FAMILY_COUNT_COLUMN] = pd.to_numeric(
    lens_aliases[LENS_FAMILY_COUNT_COLUMN],
    errors="coerce",
)


# =========================================================
# Check DOI aliases shared by multiple Lens records
# =========================================================

alias_record_counts = (
    lens_aliases.groupby("_doi_normalized")["_lens_record_id"]
    .nunique()
)

shared_aliases = alias_record_counts[alias_record_counts > 1]

print("LENS DOI ALIAS CHECK")
print("=" * 70)
print(
    "DOI aliases used by >1 Lens record:",
    f"{len(shared_aliases):,}",
)
print()

if not shared_aliases.empty:
    print("Shared DOI aliases:")
    for doi, count in shared_aliases.items():
        print(f" - {doi}: {count:,} Lens records")
    print()


# =========================================================
# Prepare DOI-alias -> Lens metric mapping
# =========================================================

# A single Lens scholarly record can contain several DOI aliases. Every alias
# must inherit that record's patent citation count and citing-family count.
#
# If the same DOI alias unexpectedly appears in multiple Lens records, max()
# avoids summing an already aggregated Lens metric. The diagnostic above makes
# such cases visible.

lens_metrics = (
    lens_aliases[
        [
            "_doi_normalized",
            LENS_PATENT_COUNT_COLUMN,
            LENS_FAMILY_COUNT_COLUMN,
        ]
    ]
    .groupby("_doi_normalized", as_index=False)
    .agg(
        {
            LENS_PATENT_COUNT_COLUMN: "max",
            LENS_FAMILY_COUNT_COLUMN: "max",
        }
    )
    .rename(
        columns={
            LENS_PATENT_COUNT_COLUMN: "Patent citation count",
            LENS_FAMILY_COUNT_COLUMN: "Citing patent family count",
        }
    )
)

print("LENS PATENT METRICS")
print("=" * 70)
print("DOI-alias metric rows       :", f"{len(lens_metrics):,}")
print(
    "Missing patent counts      :",
    f"{lens_metrics['Patent citation count'].isna().sum():,}",
)
print(
    "Missing family counts      :",
    f"{lens_metrics['Citing patent family count'].isna().sum():,}",
)
print()


# =========================================================
# Select requested DOI aliases from Scopus
# =========================================================

patent_cited = scopus[
    scopus["_doi_normalized"].isin(lens_doi_set)
].copy()

matched_dois = set(
    patent_cited["_doi_normalized"].dropna()
)

unmatched_dois = lens_doi_set - matched_dois

print("SCOPUS DOI MATCHING")
print("=" * 70)
print("Requested DOI aliases :", f"{len(lens_doi_set):,}")
print("Matched DOI aliases   :", f"{len(matched_dois):,}")
print("Matched Scopus rows   :", f"{len(patent_cited):,}")
print("Unmatched DOI aliases :", f"{len(unmatched_dois):,}")
print()


# =========================================================
# Check duplicate Scopus DOI rows
# =========================================================

scopus_duplicate_mask = (
    patent_cited["_doi_normalized"].notna()
    & patent_cited["_doi_normalized"].duplicated(keep=False)
)

n_scopus_duplicate_rows = int(scopus_duplicate_mask.sum())

print("SCOPUS DUPLICATE DOI CHECK")
print("=" * 70)
print("Duplicate matched rows:", f"{n_scopus_duplicate_rows:,}")
print()

if n_scopus_duplicate_rows > 0:
    print(
        patent_cited.loc[
            scopus_duplicate_mask,
            ["Title", "DOI", "_doi_normalized"],
        ]
        .sort_values("_doi_normalized")
        .to_string(index=False)
    )
    print()

# Keep one Scopus row per exact DOI alias.
patent_cited = (
    patent_cited
    .drop_duplicates(subset=["_doi_normalized"], keep="first")
    .reset_index(drop=True)
)


# =========================================================
# Merge Lens patent-impact metrics into Scopus metadata
# =========================================================

patent_cited = patent_cited.merge(
    lens_metrics,
    on="_doi_normalized",
    how="left",
    validate="many_to_one",
)

missing_patent_metric = patent_cited["Patent citation count"].isna()
missing_family_metric = patent_cited["Citing patent family count"].isna()

print("PATENT METRIC MERGE")
print("=" * 70)
print("Publications:", f"{len(patent_cited):,}")
print(
    "With patent citation count:",
    f"{patent_cited['Patent citation count'].notna().sum():,}",
)
print(
    "Missing patent citation count:",
    f"{missing_patent_metric.sum():,}",
)
print(
    "With patent family count:",
    f"{patent_cited['Citing patent family count'].notna().sum():,}",
)
print(
    "Missing patent family count:",
    f"{missing_family_metric.sum():,}",
)
print()

if missing_patent_metric.any() or missing_family_metric.any():
    missing_metric_mask = missing_patent_metric | missing_family_metric
    print("Publications missing Lens patent metrics:")
    print(
        patent_cited.loc[
            missing_metric_mask,
            ["Title", "DOI", "_doi_normalized"],
        ].to_string(index=False)
    )
    print()


# =========================================================
# Lens-record-level Scopus coverage
# =========================================================

# This is the important validation when Lens rows may have multiple DOI aliases.
# A Lens scholarly record is considered recovered if AT LEAST ONE of its DOI
# aliases occurs in Scopus.

lens_aliases["_matched_in_scopus"] = (
    lens_aliases["_doi_normalized"].isin(matched_dois)
)

record_coverage = (
    lens_aliases.groupby("_lens_record_id")["_matched_in_scopus"]
    .any()
)

records_with_no_doi = set(
    lens.loc[lens["_doi_count"].eq(0), "_lens_record_id"]
)

covered_record_ids = set(
    record_coverage[record_coverage].index
)

all_record_ids = set(lens["_lens_record_id"])
uncovered_record_ids = all_record_ids - covered_record_ids

# Records with no DOI are included in uncovered_record_ids; report them
# separately as well.

print("LENS RECORD COVERAGE")
print("=" * 70)
print("Lens scholarly records :", f"{len(lens):,}")
print("Recovered via >=1 DOI  :", f"{len(covered_record_ids):,}")
print("Unrecovered records    :", f"{len(uncovered_record_ids):,}")
print("Records without DOI    :", f"{len(records_with_no_doi):,}")
print()

if uncovered_record_ids:
    print("Unrecovered Lens records:")

    title_candidates = [
        "citation title",
        "Citation Title",
        "Title",
        "title",
    ]
    title_column = next(
        (column for column in title_candidates if column in lens.columns),
        None,
    )

    columns_to_show = ["_lens_record_id"]
    if title_column is not None:
        columns_to_show.append(title_column)
    columns_to_show.extend([LENS_EXTERNAL_ID_COLUMN, "_doi_list"])

    print(
        lens.loc[
            lens["_lens_record_id"].isin(uncovered_record_ids),
            columns_to_show,
        ].to_string(index=False)
    )
    print()


# =========================================================
# Keep final requested columns
# =========================================================

FINAL_OUTPUT_COLUMNS = SCOPUS_OUTPUT_COLUMNS + LENS_OUTPUT_COLUMNS

patent_cited = patent_cited[FINAL_OUTPUT_COLUMNS].copy()


# =========================================================
# Sort output
# =========================================================

patent_cited = (
    patent_cited
    .sort_values(
        [
            "Patent citation count",
            "Citing patent family count",
            "Year",
        ],
        ascending=[False, False, True],
        na_position="last",
    )
    .reset_index(drop=True)
)


# =========================================================
# Save Excel
# =========================================================

patent_cited.to_excel(OUTPUT_FILE, index=False)


# =========================================================
# Final corpus validation
# =========================================================

print("FINAL PATENT-CITED SCHOLARLY CORPUS")
print("=" * 70)
print("Documents        :", f"{len(patent_cited):,}")
print("Columns          :", f"{len(patent_cited.columns):,}")
print("With abstract    :", f"{patent_cited['Abstract'].notna().sum():,}")
print("Missing abstract :", f"{patent_cited['Abstract'].isna().sum():,}")
print("Unique DOI       :", f"{patent_cited['DOI'].nunique():,}")
print(
    "Patent counts    :",
    f"{patent_cited['Patent citation count'].notna().sum():,}",
)
print(
    "Family counts    :",
    f"{patent_cited['Citing patent family count'].notna().sum():,}",
)
print()


# =========================================================
# Patent citation statistics
# =========================================================

print("PATENT CITATION STATISTICS")
print("=" * 70)
print(
    "Total patent citations :",
    f"{patent_cited['Patent citation count'].sum():,.0f}",
)
print(
    "Mean per publication   :",
    f"{patent_cited['Patent citation count'].mean():.2f}",
)
print(
    "Median per publication :",
    f"{patent_cited['Patent citation count'].median():.2f}",
)
print(
    "Maximum                :",
    f"{patent_cited['Patent citation count'].max():,.0f}",
)
print()
print(
    "Total citing families  :",
    f"{patent_cited['Citing patent family count'].sum():,.0f}",
)
print(
    "Mean families/article  :",
    f"{patent_cited['Citing patent family count'].mean():.2f}",
)
print(
    "Median families/article:",
    f"{patent_cited['Citing patent family count'].median():.2f}",
)
print(
    "Maximum families       :",
    f"{patent_cited['Citing patent family count'].max():,.0f}",
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
    .head(20)
    .copy()
)

top_patent_cited.insert(
    0,
    "Rank",
    range(1, len(top_patent_cited) + 1),
)

print(top_patent_cited.to_string(index=False))
print()


# =========================================================
# Save confirmation
# =========================================================

print("OUTPUT")
print("=" * 70)
print("Saved:", OUTPUT_FILE)
print()


# =========================================================
# Final checks
# =========================================================

errors = []
warnings = []

# The DOI TXT and Lens export should normally contain the same alias set.
if doi_list_only:
    warnings.append(
        f"{len(doi_list_only):,} DOI(s) occur in the DOI list but not "
        "in the current Lens export."
    )

if lens_export_only:
    warnings.append(
        f"{len(lens_export_only):,} DOI alias(es) occur in the Lens export "
        "but not in the DOI list. Re-run extract_lens_dois.py."
    )

# Unmatched aliases are informational/warnings, not automatically failures.
# A Lens record can have several aliases and only one needs to be represented
# in Scopus.
if unmatched_dois:
    warnings.append(
        f"{len(unmatched_dois):,} DOI alias(es) were not found in Scopus."
    )

# Record-level coverage is the substantive completeness check.
if uncovered_record_ids:
    errors.append(
        f"{len(uncovered_record_ids):,} Lens scholarly record(s) were not "
        "recovered from Scopus through any of their DOI aliases."
    )

if patent_cited["DOI"].nunique() != len(patent_cited):
    errors.append("Final output contains duplicate DOI values.")

if patent_cited["Patent citation count"].isna().any():
    errors.append("Some publications are missing Patent citation count.")

if patent_cited["Citing patent family count"].isna().any():
    errors.append("Some publications are missing Citing patent family count.")


# =========================================================
# Final status
# =========================================================

if errors:
    print("CHECK WARNING")
    print("=" * 70)

    for error in errors:
        print(" - ERROR:", error)

    for warning in warnings:
        print(" - NOTE :", warning)

else:
    print("CHECK PASSED")
    print("=" * 70)
    print(
        "Every Lens scholarly record was recovered from Scopus through "
        "at least one DOI alias."
    )
    print(
        "All matched Scopus publications were assigned Lens patent "
        "citation and patent-family metrics."
    )
    print("The final output contains unique Scopus DOI values.")

    if warnings:
        print()
        print("NOTES")
        print("=" * 70)
        for warning in warnings:
            print(" -", warning)


# =========================================================
# Unmatched DOI aliases
# =========================================================

if unmatched_dois:
    print()
    print("UNMATCHED DOI ALIASES")
    print("=" * 70)
    for doi in sorted(unmatched_dois):
        print(" -", doi)
