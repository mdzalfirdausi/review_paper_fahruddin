#!/usr/bin/env python3

# ============================================================
# Final LDA model for the full Scopus academic ML-OPF corpus
#
# Selected model:
#   K = 50
#
# Purpose:
#   Re-estimate the selected topic model using the complete
#   retained academic corpus after model selection.
#
# Outputs:
#   1. Model summary
#   2. Topic-term posterior (beta)
#   3. Top terms
#   4. Document-topic posterior (theta)
#   5. Probability-weighted topic prevalence
#   6. Representative documents
#
# Designed for:
#   issnode1
# ============================================================


from pathlib import Path

import os
import subprocess
import sys
import time

import pandas as pd


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

SCRIPT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PROJECT_DIR = (
    SCRIPT_DIR.parent
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "output"
)

LANDSCAPE_DIR = (
    OUTPUT_DIR
    / "scopus_research_landscape"
)

LDA_DIR = (
    LANDSCAPE_DIR
    / "lda"
)

R_PREPROCESS_DIR = (
    LDA_DIR
    / "r_preprocessing"
)

FINAL_DIR = (
    LDA_DIR
    / "final_model"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# RSCRIPT
# ============================================================

RSCRIPT = Path(
    r"/nfs/mfirdausi/miniconda3/envs/pytorch/bin/Rscript"
)

if not RSCRIPT.exists():

    raise FileNotFoundError(
        f"Rscript not found:\n"
        f"{RSCRIPT}"
    )


# ============================================================
# REQUIRED R PACKAGES
# ============================================================

print(
    "Checking R environment..."
)

print()


r_package_check = subprocess.run(
    [
        str(RSCRIPT),
        "-e",
        (
            'pkgs <- c('
            '"tm", '
            '"SnowballC", '
            '"slam", '
            '"topicmodels"'
            '); '
            'ok <- sapply('
            'pkgs, '
            'requireNamespace, '
            'quietly=TRUE'
            '); '
            'cat('
            'paste(pkgs, ok, sep="="), '
            'sep="\\n"'
            ')'
        ),
    ],
    capture_output=True,
    text=True,
    check=True,
)


print("Rscript:")
print(RSCRIPT)

print()

print("Required R packages:")
print(r_package_check.stdout)


# ============================================================
# INPUT FILES
# ============================================================

PROCESSED_FILE = (
    R_PREPROCESS_DIR
    / "scopus_academic_processed.csv"
)

ACADEMIC_CORPUS_FILE = (
    LDA_DIR
    / "scopus_academic_corpus.csv"
)


# ============================================================
# FINAL OUTPUT FILES
# ============================================================

MODEL_SUMMARY_FILE = (
    LDA_DIR
    / "scopus_academic_final_model_summary.csv"
)

BETA_FILE = (
    FINAL_DIR
    / "scopus_academic_final_beta.csv"
)

TOP_TERMS_FILE = (
    LDA_DIR
    / "scopus_academic_final_top_terms.csv"
)

THETA_FILE = (
    LDA_DIR
    / "scopus_academic_final_document_topics.csv"
)

PREVALENCE_FILE = (
    LDA_DIR
    / "scopus_academic_final_topic_prevalence.csv"
)

REPRESENTATIVE_FILE = (
    LDA_DIR
    / "scopus_academic_final_representative_documents.csv"
)

LOG_FILE = (
    LDA_DIR
    / "scopus_academic_final_lda.log"
)


# ============================================================
# FINAL MODEL CONFIGURATION
# ============================================================

FINAL_K = 50

MIN_DOC_FREQ = 83

RANDOM_SEED = 123

BURN_IN = 500

ITERATIONS = 2000

THIN = 50

TOP_N = 10

REPRESENTATIVE_N = 5


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

print()

print(
    "Checking input files..."
)

print()


required_files = [
    PROCESSED_FILE,
    ACADEMIC_CORPUS_FILE,
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required input file not found:\n"
            f"{file_path}"
        )

    print(
        "Found:",
        file_path
    )


# ============================================================
# VALIDATE DOCUMENT ALIGNMENT
#
# processed_file and academic_corpus_file must describe the
# same documents in the same order.
# ============================================================

processed_check = pd.read_csv(
    PROCESSED_FILE,
    usecols=[
        "Document_ID",
    ],
)

corpus_check = pd.read_csv(
    ACADEMIC_CORPUS_FILE,
    usecols=[
        "Document_ID",
    ],
)


if len(processed_check) != len(corpus_check):

    raise ValueError(
        "Processed corpus and academic corpus contain "
        "different numbers of documents."
    )


if not processed_check[
    "Document_ID"
].equals(
    corpus_check[
        "Document_ID"
    ]
):

    raise ValueError(
        "Document_ID ordering differs between the processed "
        "corpus and academic metadata file."
    )


print()

print(
    "Document alignment verified:",
    f"{len(processed_check):,}",
    "documents"
)


# ============================================================
# CREATE FINAL R WORKER
# ============================================================

R_WORKER_SCRIPT = (
    FINAL_DIR
    / "fit_final_scopus_lda.R"
)


R_WORKER_CODE = r'''
args <- commandArgs(
    trailingOnly = TRUE
)

processed_file      <- args[1]
corpus_file         <- args[2]

summary_file        <- args[3]
beta_file           <- args[4]
top_terms_file      <- args[5]
theta_file          <- args[6]
prevalence_file     <- args[7]
representative_file <- args[8]

k                   <- as.integer(args[9])
min_doc_freq        <- as.integer(args[10])

random_seed         <- as.integer(args[11])
burn_in             <- as.integer(args[12])
iterations          <- as.integer(args[13])
thin                <- as.integer(args[14])

top_n               <- as.integer(args[15])
representative_n    <- as.integer(args[16])


suppressPackageStartupMessages({
    library(tm)
    library(SnowballC)
    library(slam)
    library(topicmodels)
})


# ============================================================
# CONFIGURATION
# ============================================================

MIN_TERM_LENGTH <- 3


# ============================================================
# LOAD PROCESSED CORPUS
# ============================================================

data <- read.csv(
    processed_file,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    fileEncoding = "UTF-8"
)


if (
    !"Document_ID"
    %in%
    names(data)
) {

    stop(
        "Document_ID column not found in processed corpus."
    )
}


if (
    !"processed_text"
    %in%
    names(data)
) {

    stop(
        "processed_text column not found in processed corpus."
    )
}


processed <- data$processed_text

processed[
    is.na(processed)
] <- ""


# ============================================================
# LOAD ORIGINAL ACADEMIC METADATA
# ============================================================

metadata <- read.csv(
    corpus_file,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    fileEncoding = "UTF-8"
)


if (
    !"Document_ID"
    %in%
    names(metadata)
) {

    stop(
        "Document_ID column not found in academic corpus."
    )
}


if (
    nrow(metadata)
    !=
    nrow(data)
) {

    stop(
        "Academic metadata and processed corpus have different numbers of rows."
    )
}


if (
    !all(
        metadata$Document_ID
        ==
        data$Document_ID
    )
) {

    stop(
        "Document_ID ordering differs between metadata and processed corpus."
    )
}


# ============================================================
# CONSTRUCT FULL-CORPUS DTM
# ============================================================

cat(
    "Constructing full-corpus DTM...\n"
)


flush.console()


corpus <- VCorpus(
    VectorSource(
        processed
    )
)


dtm <- DocumentTermMatrix(
    corpus,
    control = list(
        wordLengths = c(
            MIN_TERM_LENGTH,
            Inf
        )
    )
)


cat(
    "Raw documents:",
    nrow(dtm),
    "\n"
)


cat(
    "Raw vocabulary:",
    ncol(dtm),
    "\n"
)


# ============================================================
# APPLY FINAL DOCUMENT-FREQUENCY FILTER
# ============================================================

document_frequency <- (
    slam::col_sums(
        dtm > 0
    )
)


dtm <- dtm[
    ,
    document_frequency >= min_doc_freq
]


cat(
    "Filtered vocabulary:",
    ncol(dtm),
    "\n"
)


cat(
    "Minimum DF:",
    min_doc_freq,
    "\n"
)


# ============================================================
# REMOVE EMPTY DOCUMENTS AFTER FINAL DTM FILTERING
#
# Normally this should be zero, but the script handles it
# explicitly to preserve alignment.
# ============================================================

nonempty <- (
    slam::row_sums(
        dtm
    ) > 0
)


n_empty_removed <- sum(
    !nonempty
)


if (
    n_empty_removed > 0
) {

    dtm <- dtm[
        nonempty,
    ]


    metadata <- metadata[
        nonempty,
        ,
        drop = FALSE
    ]


    data <- data[
        nonempty,
        ,
        drop = FALSE
    ]
}


cat(
    "Documents retained:",
    nrow(dtm),
    "\n"
)


cat(
    "Empty documents removed:",
    n_empty_removed,
    "\n"
)


# ============================================================
# FINAL MODEL DIAGNOSTICS
# ============================================================

cat(
    "Final K:",
    k,
    "\n"
)


cat(
    "Random seed:",
    random_seed,
    "\n"
)


cat(
    "Burn-in:",
    burn_in,
    "\n"
)


cat(
    "Iterations:",
    iterations,
    "\n"
)


cat(
    "Thin:",
    thin,
    "\n"
)


cat(
    "Top terms:",
    top_n,
    "\n"
)


cat(
    "Representative documents per topic:",
    representative_n,
    "\n\n"
)


flush.console()


# ============================================================
# FIT FINAL LDA MODEL
# ============================================================

cat(
    "Fitting final K =",
    k,
    "LDA model on full corpus...\n"
)


flush.console()


start_time <- Sys.time()


model <- topicmodels::LDA(
    dtm,
    k = k,
    method = "Gibbs",
    control = list(
        seed = random_seed,
        burnin = burn_in,
        iter = iterations,
        thin = thin
    )
)


# ============================================================
# POSTERIOR DISTRIBUTIONS
#
# beta  = topic x term
# theta = document x topic
# ============================================================

cat(
    "Extracting posterior distributions...\n"
)


flush.console()


posterior_result <- posterior(
    model
)


beta <- posterior_result$terms

theta <- posterior_result$topics


if (
    nrow(beta)
    !=
    k
) {

    stop(
        "Unexpected beta dimensions."
    )
}


if (
    ncol(theta)
    !=
    k
) {

    stop(
        "Unexpected theta dimensions."
    )
}


if (
    nrow(theta)
    !=
    nrow(metadata)
) {

    stop(
        "Theta rows do not match metadata rows."
    )
}


# ============================================================
# SAVE FULL BETA MATRIX
#
# Long format:
#   K, Topic, Term, Beta
# ============================================================

cat(
    "Saving topic-term posterior...\n"
)


flush.console()


beta_results <- data.frame()


for (
    topic in seq_len(k)
) {

    topic_beta <- data.frame(

        K = rep(
            k,
            ncol(beta)
        ),

        Topic = rep(
            topic,
            ncol(beta)
        ),

        Term = colnames(
            beta
        ),

        Beta = as.numeric(
            beta[
                topic,
            ]
        ),

        stringsAsFactors = FALSE
    )


    beta_results <- rbind(
        beta_results,
        topic_beta
    )
}


write.csv(
    beta_results,
    beta_file,
    row.names = FALSE
)


# ============================================================
# TOP TERMS
# ============================================================

cat(
    "Extracting top terms...\n"
)


flush.console()


top_term_results <- data.frame()


for (
    topic in seq_len(k)
) {

    top_indices <- order(
        beta[
            topic,
        ],
        decreasing = TRUE
    )[
        seq_len(
            min(
                top_n,
                ncol(beta)
            )
        )
    ]


    top_words <- colnames(
        beta
    )[
        top_indices
    ]


    top_beta <- beta[
        topic,
        top_indices
    ]


    topic_terms <- data.frame(

        K = rep(
            k,
            length(top_words)
        ),

        Topic = rep(
            topic,
            length(top_words)
        ),

        Rank = seq_along(
            top_words
        ),

        Term = top_words,

        Beta = as.numeric(
            top_beta
        ),

        stringsAsFactors = FALSE
    )


    top_term_results <- rbind(
        top_term_results,
        topic_terms
    )
}


write.csv(
    top_term_results,
    top_terms_file,
    row.names = FALSE
)


# ============================================================
# DOCUMENT-TOPIC POSTERIOR
#
# Wide format:
#   Document_ID, Topic_01, ..., Topic_50
# ============================================================

cat(
    "Saving document-topic posterior...\n"
)


flush.console()


theta_results <- as.data.frame(
    theta
)


topic_names <- sprintf(
    "Topic_%02d",
    seq_len(k)
)


names(
    theta_results
) <- topic_names


theta_results <- cbind(

    Document_ID =
        metadata$Document_ID,

    theta_results
)


write.csv(
    theta_results,
    theta_file,
    row.names = FALSE
)


# ============================================================
# PROBABILITY-WEIGHTED TOPIC PREVALENCE
#
# prevalence_k =
#   mean_d theta[d,k]
#
# Since each document's theta sums to 1, topic prevalence
# across topics also sums to 1.
# ============================================================

cat(
    "Calculating topic prevalence...\n"
)


flush.console()


topic_prevalence <- colMeans(
    theta
)


prevalence_results <- data.frame(

    K = rep(
        k,
        k
    ),

    Topic = seq_len(
        k
    ),

    Mean_Probability =
        as.numeric(
            topic_prevalence
        ),

    Prevalence_Percent =
        100
        *
        as.numeric(
            topic_prevalence
        ),

    stringsAsFactors = FALSE
)


prevalence_results <- (
    prevalence_results[
        order(
            prevalence_results$Prevalence_Percent,
            decreasing = TRUE
        ),
    ]
)


prevalence_results$Prevalence_Rank <- (
    seq_len(
        nrow(
            prevalence_results
        )
    )
)


prevalence_results <- prevalence_results[
    ,
    c(
        "K",
        "Topic",
        "Prevalence_Rank",
        "Mean_Probability",
        "Prevalence_Percent"
    )
]


write.csv(
    prevalence_results,
    prevalence_file,
    row.names = FALSE
)


# ============================================================
# REPRESENTATIVE DOCUMENTS
#
# For each topic, select documents with the largest theta.
# ============================================================

cat(
    "Selecting representative documents...\n"
)


flush.console()


representative_results <- data.frame()


for (
    topic in seq_len(k)
) {

    topic_probability <- (
        theta[
            ,
            topic
        ]
    )


    representative_indices <- order(
        topic_probability,
        decreasing = TRUE
    )[
        seq_len(
            min(
                representative_n,
                length(
                    topic_probability
                )
            )
        )
    ]


    topic_documents <- data.frame(

        K = rep(
            k,
            length(
                representative_indices
            )
        ),

        Topic = rep(
            topic,
            length(
                representative_indices
            )
        ),

        Rank = seq_along(
            representative_indices
        ),

        Document_ID =
            metadata$Document_ID[
                representative_indices
            ],

        Topic_Probability =
            topic_probability[
                representative_indices
            ],

        stringsAsFactors = FALSE
    )


    # --------------------------------------------------------
    # Add metadata when available
    # --------------------------------------------------------

    if (
        "Title"
        %in%
        names(metadata)
    ) {

        topic_documents$Title <- (
            metadata$Title[
                representative_indices
            ]
        )
    }


    if (
        "Year"
        %in%
        names(metadata)
    ) {

        topic_documents$Year <- (
            metadata$Year[
                representative_indices
            ]
        )
    }


    if (
        "DOI"
        %in%
        names(metadata)
    ) {

        topic_documents$DOI <- (
            metadata$DOI[
                representative_indices
            ]
        )
    }


    if (
        "Abstract"
        %in%
        names(metadata)
    ) {

        topic_documents$Abstract <- (
            metadata$Abstract[
                representative_indices
            ]
        )
    }


    representative_results <- rbind(
        representative_results,
        topic_documents
    )
}


write.csv(
    representative_results,
    representative_file,
    row.names = FALSE
)


# ============================================================
# FINAL MODEL SUMMARY
# ============================================================

elapsed <- as.numeric(
    difftime(
        Sys.time(),
        start_time,
        units = "secs"
    )
)


# ------------------------------------------------------------
# Check posterior normalization
# ------------------------------------------------------------

theta_row_sums <- rowSums(
    theta
)


beta_row_sums <- rowSums(
    beta
)


theta_max_error <- max(
    abs(
        theta_row_sums
        - 1
    )
)


beta_max_error <- max(
    abs(
        beta_row_sums
        - 1
    )
)


prevalence_sum <- sum(
    topic_prevalence
)


summary_results <- data.frame(

    K = k,

    Documents =
        nrow(dtm),

    Vocabulary =
        ncol(dtm),

    Minimum_DF =
        min_doc_freq,

    Empty_Documents_Removed =
        n_empty_removed,

    Random_Seed =
        random_seed,

    Burn_In =
        burn_in,

    Iterations =
        iterations,

    Thin =
        thin,

    Top_N =
        top_n,

    Representative_N =
        representative_n,

    Theta_Max_RowSum_Error =
        theta_max_error,

    Beta_Max_RowSum_Error =
        beta_max_error,

    Topic_Prevalence_Sum =
        prevalence_sum,

    Elapsed_seconds =
        elapsed
)


write.csv(
    summary_results,
    summary_file,
    row.names = FALSE
)


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

cat(
    "\n"
)


cat(
    "============================================================\n"
)


cat(
    "FINAL SCOPUS ACADEMIC LDA COMPLETE\n"
)


cat(
    "============================================================\n"
)


cat(
    "K:",
    k,
    "\n"
)


cat(
    "Documents:",
    nrow(dtm),
    "\n"
)


cat(
    "Vocabulary:",
    ncol(dtm),
    "\n"
)


cat(
    "Theta max row-sum error:",
    theta_max_error,
    "\n"
)


cat(
    "Beta max row-sum error:",
    beta_max_error,
    "\n"
)


cat(
    "Topic prevalence sum:",
    prevalence_sum,
    "\n"
)


cat(
    "Elapsed seconds:",
    elapsed,
    "\n"
)


cat(
    "============================================================\n"
)


flush.console()
'''


R_WORKER_SCRIPT.write_text(
    R_WORKER_CODE,
    encoding="utf-8",
)


print()

print(
    "Created final R worker:",
    R_WORKER_SCRIPT
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print()

print(
    "=" * 76
)

print(
    "FINAL SCOPUS ACADEMIC LDA"
)

print(
    "=" * 76
)

print(
    "Project directory       :",
    PROJECT_DIR
)

print(
    "Processed corpus        :",
    PROCESSED_FILE
)

print(
    "Academic metadata       :",
    ACADEMIC_CORPUS_FILE
)

print(
    "Final K                 :",
    FINAL_K
)

print(
    "Minimum DF              :",
    MIN_DOC_FREQ
)

print(
    "Random seed             :",
    RANDOM_SEED
)

print(
    "Burn-in                 :",
    BURN_IN
)

print(
    "Iterations              :",
    ITERATIONS
)

print(
    "Thin                    :",
    THIN
)

print(
    "Top terms               :",
    TOP_N
)

print(
    "Representative documents:",
    REPRESENTATIVE_N
)

print(
    "=" * 76
)


# ============================================================
# R COMMAND
# ============================================================

command = [
    str(RSCRIPT),
    str(R_WORKER_SCRIPT),

    str(PROCESSED_FILE),
    str(ACADEMIC_CORPUS_FILE),

    str(MODEL_SUMMARY_FILE),
    str(BETA_FILE),
    str(TOP_TERMS_FILE),
    str(THETA_FILE),
    str(PREVALENCE_FILE),
    str(REPRESENTATIVE_FILE),

    str(FINAL_K),
    str(MIN_DOC_FREQ),

    str(RANDOM_SEED),
    str(BURN_IN),
    str(ITERATIONS),
    str(THIN),

    str(TOP_N),
    str(REPRESENTATIVE_N),
]


# ============================================================
# ENVIRONMENT
#
# Keep one R process single-threaded at the BLAS/OpenMP level.
# topicmodels performs the Gibbs work itself.
# ============================================================

env = os.environ.copy()

env[
    "OMP_NUM_THREADS"
] = "1"

env[
    "OPENBLAS_NUM_THREADS"
] = "1"

env[
    "MKL_NUM_THREADS"
] = "1"

env[
    "NUMEXPR_NUM_THREADS"
] = "1"

env[
    "VECLIB_MAXIMUM_THREADS"
] = "1"


# ============================================================
# RUN FINAL MODEL
# ============================================================

print()

print(
    "Running final K=50 model..."
)

print(
    "Log:",
    LOG_FILE
)

print()


wall_start = (
    time.time()
)


with LOG_FILE.open(
    "w",
    encoding="utf-8",
) as log:

    process = subprocess.run(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )


wall_elapsed = (
    time.time()
    - wall_start
)


# ============================================================
# VALIDATE EXECUTION
# ============================================================

if process.returncode != 0:

    print(
        "Final LDA model failed."
    )

    print(
        "Inspect:",
        LOG_FILE
    )

    sys.exit(
        process.returncode
    )


required_outputs = [
    MODEL_SUMMARY_FILE,
    BETA_FILE,
    TOP_TERMS_FILE,
    THETA_FILE,
    PREVALENCE_FILE,
    REPRESENTATIVE_FILE,
]


missing_outputs = [
    path
    for path
    in required_outputs
    if not path.exists()
]


if missing_outputs:

    print(
        "Final LDA completed but required "
        "outputs are missing:"
    )

    for path in missing_outputs:

        print(
            " -",
            path
        )

    sys.exit(1)


# ============================================================
# LOAD AND VALIDATE RESULTS IN PYTHON
# ============================================================

summary = pd.read_csv(
    MODEL_SUMMARY_FILE
)

top_terms = pd.read_csv(
    TOP_TERMS_FILE
)

theta = pd.read_csv(
    THETA_FILE
)

prevalence = pd.read_csv(
    PREVALENCE_FILE
)

representative = pd.read_csv(
    REPRESENTATIVE_FILE
)


# ------------------------------------------------------------
# Expected dimensions
# ------------------------------------------------------------

expected_top_term_rows = (
    FINAL_K
    * TOP_N
)

expected_representative_rows = (
    FINAL_K
    * REPRESENTATIVE_N
)


if len(top_terms) != expected_top_term_rows:

    raise ValueError(
        "Unexpected number of top-term rows: "
        f"{len(top_terms):,}; expected "
        f"{expected_top_term_rows:,}."
    )


if len(prevalence) != FINAL_K:

    raise ValueError(
        "Unexpected number of prevalence rows: "
        f"{len(prevalence):,}; expected "
        f"{FINAL_K:,}."
    )


if len(representative) != expected_representative_rows:

    raise ValueError(
        "Unexpected number of representative-document rows: "
        f"{len(representative):,}; expected "
        f"{expected_representative_rows:,}."
    )


expected_theta_columns = (
    FINAL_K
    + 1
)


if theta.shape[1] != expected_theta_columns:

    raise ValueError(
        "Unexpected number of theta columns: "
        f"{theta.shape[1]:,}; expected "
        f"{expected_theta_columns:,}."
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()

print(
    "=" * 76
)

print(
    "FINAL SCOPUS ACADEMIC LDA COMPLETED SUCCESSFULLY"
)

print(
    "=" * 76
)

print()

print(
    "Wall time:",
    f"{wall_elapsed / 60:.2f} minutes"
)

print()

print(
    "Model summary:"
)

print(
    summary.to_string(
        index=False
    )
)

print()

print(
    "Top-term rows:",
    f"{len(top_terms):,}"
)

print(
    "Theta dimensions:",
    f"{theta.shape[0]:,} x "
    f"{theta.shape[1]:,}"
)

print(
    "Prevalence rows:",
    f"{len(prevalence):,}"
)

print(
    "Representative-document rows:",
    f"{len(representative):,}"
)

print()

print(
    "Output files:"
)

for path in required_outputs:

    print(
        " -",
        path
    )

print()

print(
    "Final model log:"
)

print(
    " -",
    LOG_FILE
)