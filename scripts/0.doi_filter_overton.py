import re
from pathlib import Path

import pandas as pd


CSV_FILE = Path(
    "data/overton_policy_doc_2026-09-10_12-24-30.csv"
)
COLUMN = "Cited DOIs"

# Recommended for Scopus Advanced Search.
# Larger values may cause "query too complex" errors.
BATCH_SIZE = 1000


# Exact repairs confirmed from your data.
EXACT_REPAIRS = {
    "10.48550/arxiv.2303.10130on5april2024)":
        "10.48550/arxiv.2303.10130",

    "10.1787/ab43a9a5-en29april2021)":
        "10.1787/ab43a9a5-en",

    "10.5281/zenodo.198183).themainassumptionsareasfollows":
        "10.5281/zenodo.198183",

    "10.5281/zenodo.198181).thek-functionisdefinedforaspatio-temporalstochasticprocessas":
        "10.5281/zenodo.198181",

    "10.5194/hess-24-4109-20204109-2020,2020":
        "10.5194/hess-24-4109-2020",

    "10.1007/978-1-4471-6302-2{_}5":
        "10.1007/978-1-4471-6302-2_5",

    # Confirmed by your DOI search
    "10.4172/2332-0915.100015118,2014":
        "10.4172/2332-0915.100015118",

    # Confirmed by your DOI search
    "10.36227/techrxiv.19690570.v14.detailsoftheimpact(indicativemaximum750words)":
        "10.36227/techrxiv.19690570",
}


# Permitted characters for modern and legacy DOI formats.
# This allows balanced (), [] and <> characters.
DOI_PATTERN = re.compile(
    r"^10\.\d{4,9}/[-._;()/:a-z0-9<>\[\]]+$",
    flags=re.IGNORECASE,
)


# Text indicating that citation metadata was appended to a DOI.
METADATA_MARKERS = (
    "activity:",
    "availableat",
    "copiesof",
    "retrievedfrom",
    "scopuscitations",
    "numberofauthors",
    "organisedby",
    "published:",
    "libraryofcongress",
    "informationon",
    "detailsoftheimpact",
    "themainassumptions",
    "k-function",
    "ontheconceptof",
    ".veracylinder",
    ".nypa(",
    ".---.",
    ".short",
    "/full",
)


# Patterns for extracting valid DOI prefixes from contaminated values.
# More specific patterns should appear before general patterns.
RECOVERY_PATTERNS = [
    # OMICS
    re.compile(
        r"^(10\.4172/\d{4}-\d{4}\.\d+)",
        re.IGNORECASE,
    ),

    # TechRxiv
    re.compile(
        r"^(10\.36227/techrxiv\.\d+)",
        re.IGNORECASE,
    ),

    # Elsevier journal DOI:
    # 10.1016/j.energy.2020.117188
    # 10.1016/j.energy.2017.12.051
    re.compile(
        r"^(10\.1016/j\.[a-z0-9]+"
        r"\.\d{4}\.(?:\d{2}\.)?\d+)",
        re.IGNORECASE,
    ),

    # MDPI
    re.compile(
        r"^(10\.3390/[a-z]+\d+)",
        re.IGNORECASE,
    ),

    # IEEE journal and conference papers
    re.compile(
        r"^(10\.1109/[a-z0-9-]+\.\d{4}\.\d+)",
        re.IGNORECASE,
    ),

    # National Academies Press
    re.compile(
        r"^(10\.17226/\d+)",
        re.IGNORECASE,
    ),

    # OSTI reports
    re.compile(
        r"^(10\.2172/\d+)",
        re.IGNORECASE,
    ),

    # arXiv
    re.compile(
        r"^(10\.48550/arxiv\.\d{4}\.\d{4,5})",
        re.IGNORECASE,
    ),

    # Zenodo
    re.compile(
        r"^(10\.5281/zenodo\.\d+)",
        re.IGNORECASE,
    ),

    # SSRN
    re.compile(
        r"^(10\.2139/ssrn\.\d+)",
        re.IGNORECASE,
    ),

    # SIAM
    re.compile(
        r"^(10\.1137/\d+)",
        re.IGNORECASE,
    ),

    # American Chemical Society
    re.compile(
        r"^(10\.1021/[a-z0-9]+)",
        re.IGNORECASE,
    ),

    # SPIE
    re.compile(
        r"^(10\.1117/12\.\d+)",
        re.IGNORECASE,
    ),

    # Nordic journal platform
    re.compile(
        r"^(10\.5324/[a-z0-9.]+\.\d+)",
        re.IGNORECASE,
    ),

    # US Geological Survey
    re.compile(
        r"^(10\.3133/[a-z]+\d+)",
        re.IGNORECASE,
    ),

    # Fourth National Climate Assessment
    re.compile(
        r"^(10\.7930/nca4\.2018)",
        re.IGNORECASE,
    ),

    # Taylor & Francis numeric DOI
    re.compile(
        r"^(10\.1080/\d+)",
        re.IGNORECASE,
    ),

    # Cambridge book DOI
    re.compile(
        r"^(10\.1017/\d+)",
        re.IGNORECASE,
    ),

    # IOP
    re.compile(
        r"^(10\.1088/\d{4}-\d{4}/[a-z0-9]+)",
        re.IGNORECASE,
    ),

    # Older Nature DOI
    re.compile(
        r"^(10\.1038/\d+[a-z])",
        re.IGNORECASE,
    ),

    # European Commission publications
    re.compile(
        r"^(10\.2861/\d+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(10\.2760/\d+)",
        re.IGNORECASE,
    ),

    # OECD publications
    re.compile(
        r"^(10\.1787/[a-z0-9]+-en)",
        re.IGNORECASE,
    ),

    # Frontiers
    re.compile(
        r"^(10\.3389/[a-z]+\.\d+\.\d+)",
        re.IGNORECASE,
    ),

    # Science and Science Advances
    re.compile(
        r"^(10\.1126/[a-z]+\.[a-z0-9]+)",
        re.IGNORECASE,
    ),

    # American Geophysical Union
    re.compile(
        r"^(10\.1029/[a-z0-9]+)",
        re.IGNORECASE,
    ),

    # Springer journal DOI
    re.compile(
        r"^(10\.1007/s\d{5}-\d{3}-\d{5}-[a-z0-9])",
        re.IGNORECASE,
    ),

    # RISJ/DataCite-style identifier
    re.compile(
        r"^(10\.60625/[a-z0-9]{4}"
        r"(?:-[a-z0-9]{4}){2})",
        re.IGNORECASE,
    ),

    # Wiley book DOI
    re.compile(
        r"^(10\.1002/\d+)",
        re.IGNORECASE,
    ),

    # ESTS
    re.compile(
        r"^(10\.17351/ests\d+\.\d+)",
        re.IGNORECASE,
    ),

    # Oxford books and book chapters
    re.compile(
        r"^(10\.1093/acprof:oso/\d+\.\d+\.\d+)",
        re.IGNORECASE,
    ),

    # DEPA
    re.compile(
        r"^(10\.17196/[a-z]+\.\d+)",
        re.IGNORECASE,
    ),

    # Emerald
    re.compile(
        r"^(10\.1108/[a-z0-9-]+)",
        re.IGNORECASE,
    ),

    # Copernicus publications
    re.compile(
        r"^(10\.5194/[a-z]+-\d+(?:-\d+)+)",
        re.IGNORECASE,
    ),
]


def normalize_doi_text(value):
    """Apply basic normalization without guessing the DOI endpoint."""
    value = str(value).strip().lower()

    value = re.sub(
        r"^(?:doi:\s*|https?://(?:dx\.)?doi\.org/)",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = value.strip(" \t\r\n\"'")

    return value


def is_clean_doi(value):
    """Return True when the complete value looks like a valid DOI."""
    if not value:
        return False

    if not value.isascii():
        return False

    if any(character.isspace() for character in value):
        return False

    if any(marker in value for marker in METADATA_MARKERS):
        return False

    # Check all bracket types independently.
    if value.count("(") != value.count(")"):
        return False

    if value.count("[") != value.count("]"):
        return False

    if value.count("<") != value.count(">"):
        return False

    return DOI_PATTERN.fullmatch(value) is not None


def clean_doi(value):
    """
    Normalize and extract one DOI.

    If citation text is appended to the DOI, attempt to recover the
    valid DOI prefix. Return None only when no reliable DOI is found.
    """
    value = normalize_doi_text(value)

    # Find a DOI even when other text appears before it.
    doi_start = re.search(
        r"10\.\d{4,9}/",
        value,
        flags=re.IGNORECASE,
    )

    if not doi_start:
        return None

    value = value[doi_start.start():]

    # Ignore final citation punctuation when checking exact repairs.
    repair_key = value.rstrip(".;")

    if repair_key in EXACT_REPAIRS:
        return EXACT_REPAIRS[repair_key]

    # Remove common final citation punctuation.
    candidate = value.rstrip(".,;")

    # Return unchanged when the complete DOI is already clean.
    if is_clean_doi(candidate):
        return candidate

    # Recover a valid publisher-specific DOI prefix.
    for pattern in RECOVERY_PATTERNS:
        match = pattern.match(value)

        if not match:
            continue

        candidate = match.group(1).rstrip(".,;")

        if is_clean_doi(candidate):
            return candidate

    return None


df = pd.read_csv(
    CSV_FILE,
    dtype=str,
    encoding="utf-8-sig",
)

if COLUMN not in df.columns:
    raise KeyError(
        f"Column {COLUMN!r} was not found. "
        f"Available columns: {df.columns.tolist()}"
    )


dois = []
seen_dois = set()

recovered_values = []
seen_recovered = set()

rejected_values = []
seen_rejected = set()


for cell in df[COLUMN].dropna():
    # Overton commonly separates multiple values with semicolons
    # or line breaks.
    for item in re.split(r"\s*;\s*|\r?\n", cell):
        raw_value = item.strip()

        if not raw_value:
            continue

        doi = clean_doi(raw_value)

        if doi is None:
            if raw_value not in seen_rejected:
                seen_rejected.add(raw_value)
                rejected_values.append(raw_value)

            continue

        normalized_raw = normalize_doi_text(
            raw_value
        ).rstrip(".,;")

        # Record values from which a DOI was recovered.
        if doi != normalized_raw:
            recovery_record = f"{raw_value}\t=>\t{doi}"

            if recovery_record not in seen_recovered:
                seen_recovered.add(recovery_record)
                recovered_values.append(recovery_record)

        if doi not in seen_dois:
            seen_dois.add(doi)
            dois.append(doi)


print(f"Found {len(dois):,} unique DOIs.")
print(
    f"Recovered {len(recovered_values):,} "
    "DOIs from contaminated values."
)
print(
    f"Rejected {len(rejected_values):,} "
    "values without a reliable DOI."
)


# Save the final cleaned DOI list.
Path("data/cleaned_dois.txt").write_text(
    "\n".join(dois),
    encoding="utf-8",
)


# Save original-to-recovered DOI mappings.
Path("data/recovered_dois.txt").write_text(
    "\n".join(recovered_values),
    encoding="utf-8",
)


# These are usually citation fragments without any DOI.
Path("data/rejected_values.txt").write_text(
    "\n".join(rejected_values),
    encoding="utf-8",
)


# Construct Scopus Advanced Search queries.
queries = []

for start in range(0, len(dois), BATCH_SIZE):
    batch = dois[start:start + BATCH_SIZE]

    query = " OR ".join(
        f"DOI({doi})"
        for doi in batch
    )

    queries.append(query)

Path("data/scopus_doi_queries.txt").write_text(
    "\n\n".join(queries),
    encoding="utf-8",
)
Path("data/scopus_doi_queries.txt").write_text(
    "\n\n".join(queries),
    encoding="utf-8",
)


print(f"Created {len(queries):,} Scopus queries.")
print("Files created:")
print("  data/cleaned_dois.txt")
print("  data/recovered_dois.txt")
print("  data/rejected_values.txt")
print("  data/scopus_doi_queries.txt")