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
    / (
        "lenspatcite-export-M_Muna-84f6bb81-32bc-4ec9-"
        "97c6-f970293a9752-2026-09-25_01-48-07_cited.xlsx"
    )
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

print(
    "Project directory :",
    PROJECT_DIR
)

print(
    "Lens directory    :",
    LENS_DIR
)

print(
    "Input file        :",
    INPUT_FILE
)

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

# A Lens external-ID cell may contain one DOI:
#
# DOI:10.1016/j.neucom.2013.03.029
# MAGID:magid:1985230864
#
# Or it may contain multiple DOIs:
#
# DOI:10.20944/preprints202507.1980.v1
# DOI:10.3390/s25185646
# PMID:41012884
# PMCID:pmc12473902
#
# ALL DOI identifiers are extracted.
#
# The DOI stops at whitespace/newline. Other identifiers such
# as PMID, PMCID, MAGID, etc. are ignored.

DOI_PATTERN = re.compile(
    r"(?i)\bDOI:\s*"
    r"(10\.\d{4,9}/[^\s]+)"
)


def normalize_doi(value):
    """
    Normalize an extracted DOI.

    Returns
    -------
    str
        Normalized lowercase DOI.
    """

    doi = (
        str(value)
        .strip()
        .rstrip(
            ".,;"
        )
        .lower()
    )

    return doi


def extract_dois(value):
    """
    Extract ALL DOIs from a Lens 'citation external id' cell.

    Parameters
    ----------
    value : object
        Value from the Lens external-ID column.

    Returns
    -------
    list[str]
        All normalized DOI values found in the cell.

    Examples
    --------
    Input:

        DOI:10.20944/preprints202507.1980.v1
        DOI:10.3390/s25185646
        PMID:41012884
        PMCID:pmc12473902

    Output:

        [
            "10.20944/preprints202507.1980.v1",
            "10.3390/s25185646",
        ]
    """

    if pd.isna(value):

        return []


    text = str(
        value
    ).strip()


    matches = DOI_PATTERN.findall(
        text
    )


    dois = [
        normalize_doi(
            match
        )
        for match in matches
    ]


    # Remove repeated occurrences of the same DOI within
    # a single cell while preserving the original order.

    dois = list(
        dict.fromkeys(
            dois
        )
    )


    return dois


# =========================================================
# Extract all DOIs from every Lens row
# =========================================================

df = df.copy()


df["clean_dois"] = (
    df[
        EXTERNAL_ID_COLUMN
    ]
    .apply(
        extract_dois
    )
)


# Number of DOI identifiers found in each Lens row.

df["doi_count"] = (
    df[
        "clean_dois"
    ]
    .apply(
        len
    )
)


# =========================================================
# DOI diagnostics
# =========================================================

n_documents = len(
    df
)


n_with_doi = (
    df[
        "doi_count"
    ]
    .gt(
        0
    )
    .sum()
)


n_without_doi = (
    df[
        "doi_count"
    ]
    .eq(
        0
    )
    .sum()
)


n_with_one_doi = (
    df[
        "doi_count"
    ]
    .eq(
        1
    )
    .sum()
)


n_with_multiple_dois = (
    df[
        "doi_count"
    ]
    .gt(
        1
    )
    .sum()
)


n_doi_occurrences = int(
    df[
        "doi_count"
    ]
    .sum()
)


# =========================================================
# Flatten all DOI lists
# =========================================================

all_dois = [
    doi
    for row_dois in df[
        "clean_dois"
    ]
    for doi in row_dois
]


n_unique_dois = len(
    set(
        all_dois
    )
)


n_duplicate_extra = (
    len(all_dois)
    -
    n_unique_dois
)


print("DOI extraction")
print("--------------")

print(
    f"Scholar documents       : "
    f"{n_documents:,}"
)

print(
    f"Rows with DOI           : "
    f"{n_with_doi:,}"
)

print(
    f"Rows without DOI        : "
    f"{n_without_doi:,}"
)

print(
    f"Rows with one DOI       : "
    f"{n_with_one_doi:,}"
)

print(
    f"Rows with multiple DOIs : "
    f"{n_with_multiple_dois:,}"
)

print(
    f"Total DOI occurrences   : "
    f"{n_doi_occurrences:,}"
)

print(
    f"Unique DOIs             : "
    f"{n_unique_dois:,}"
)

print(
    f"Duplicate DOI excess    : "
    f"{n_duplicate_extra:,}"
)

print()


# =========================================================
# Inspect rows without DOI
# =========================================================

without_doi_mask = (
    df[
        "doi_count"
    ]
    .eq(
        0
    )
)


if without_doi_mask.any():

    print(
        "Rows without a DOI"
    )

    print(
        "------------------"
    )


    title_candidates = [
        "Citation Title",
        "citation title",
        "Title",
        "title",
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
            without_doi_mask,
            columns_to_show,
        ]
        .to_string(
            index=False
        )
    )

    print()


# =========================================================
# Inspect rows containing multiple DOIs
# =========================================================

multiple_doi_mask = (
    df[
        "doi_count"
    ]
    .gt(
        1
    )
)


if multiple_doi_mask.any():

    print(
        "Rows with multiple DOIs"
    )

    print(
        "-----------------------"
    )


    title_candidates = [
        "Citation Title",
        "citation title",
        "Title",
        "title",
    ]


    columns_to_show = [
        column
        for column in title_candidates
        if column in df.columns
    ]


    diagnostic_df = (
        df.loc[
            multiple_doi_mask,
            columns_to_show
            +
            [
                EXTERNAL_ID_COLUMN,
                "clean_dois",
            ],
        ]
        .copy()
    )


    diagnostic_df[
        "clean_dois"
    ] = (
        diagnostic_df[
            "clean_dois"
        ]
        .apply(
            lambda values:
                "; ".join(values)
        )
    )


    print(
        diagnostic_df
        .to_string(
            index=False
        )
    )

    print()


# =========================================================
# Inspect DOIs appearing in more than one Lens row
# =========================================================

doi_series = pd.Series(
    all_dois,
    dtype="object",
)


duplicate_dois = (
    doi_series[
        doi_series.duplicated(
            keep=False
        )
    ]
    .sort_values()
)


if not duplicate_dois.empty:

    print(
        "DOIs occurring more than once"
    )

    print(
        "------------------------------"
    )


    duplicate_counts = (
        duplicate_dois
        .value_counts()
        .sort_index()
    )


    for doi, count in duplicate_counts.items():

        print(
            f"{doi} : "
            f"{count:,} occurrences"
        )


    print()


# =========================================================
# Save one UNIQUE DOI per line
# =========================================================

# Preserve the order in which DOI values first occur in the
# Lens export.

dois = list(
    dict.fromkeys(
        all_dois
    )
)


OUTPUT_FILE.write_text(
    "\n".join(
        dois
    )
    +
    (
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
    f"Unique DOIs written: "
    f"{len(dois):,}"
)

print()


# =========================================================
# Validation
# =========================================================

errors = []


if n_without_doi > 0:

    errors.append(
        f"{n_without_doi:,} scholarly document(s) "
        f"do not contain an extractable DOI."
    )


if len(dois) != n_unique_dois:

    errors.append(
        "The number of written DOI values does not "
        "match the number of unique extracted DOIs."
    )


if len(dois) != len(
    set(
        dois
    )
):

    errors.append(
        "The output contains duplicate DOI values."
    )


# =========================================================
# Final status
# =========================================================

if errors:

    print(
        "CHECK WARNING"
    )

    print(
        "-------------"
    )


    for error in errors:

        print(
            " -",
            error
        )


else:

    print(
        "CHECK PASSED"
    )

    print(
        "------------"
    )


    print(
        "Every Lens scholarly document contains "
        "at least one extractable DOI."
    )

    print(
        "All DOI identifiers in each Lens row "
        "were extracted."
    )

    print(
        "The output contains one line per unique DOI."
    )


# =========================================================
# Multiple-DOI summary
# =========================================================

if n_with_multiple_dois > 0:

    print()

    print(
        "MULTIPLE-DOI NOTE"
    )

    print(
        "-----------------"
    )

    print(
        f"{n_with_multiple_dois:,} Lens document(s) "
        f"contain more than one DOI."
    )

    print(
        "All DOI values from those documents were "
        "included in the output."
    )