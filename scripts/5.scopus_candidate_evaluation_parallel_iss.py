#!/usr/bin/env python3

# ============================================================
# Parallel candidate LDA evaluation for full Scopus corpus
#
# Candidate models:
#   K = 40, 50, 60, 70
#
# Evaluation:
#   1. Held-out perplexity
#   2. Semantic coherence
#   3. Pairwise topic cosine similarity
#   4. Top-N terms per topic
#
# Designed for issnode1.
# ============================================================


from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)

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

CANDIDATE_DIR = (
    LDA_DIR
    / "candidate_parallel"
)

CANDIDATE_DIR.mkdir(
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
# CHECK R PACKAGES
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


print(
    "Rscript:"
)

print(
    RSCRIPT
)

print()

print(
    "Required R packages:"
)

print(
    r_package_check.stdout
)


# ============================================================
# INPUT FILES
# ============================================================

PROCESSED_FILE = (
    R_PREPROCESS_DIR
    / "scopus_academic_processed.csv"
)

TRAIN_FILE = (
    LDA_DIR
    / "scopus_academic_train_indices.csv"
)

TEST_FILE = (
    LDA_DIR
    / "scopus_academic_test_indices.csv"
)


# ============================================================
# FINAL OUTPUT FILES
# ============================================================

SUMMARY_FILE = (
    LDA_DIR
    / "scopus_academic_candidate_evaluation.csv"
)

TOPIC_FILE = (
    LDA_DIR
    / "scopus_academic_candidate_topic_diagnostics.csv"
)

TOP_TERMS_FILE = (
    LDA_DIR
    / "scopus_academic_candidate_top_terms.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

K_VALUES = [40, 50, 60, 70, 80, 90]


MIN_DOC_FREQ = 254

RANDOM_SEED = 123


# ------------------------------------------------------------
# Longer Gibbs run than coarse screening
# ------------------------------------------------------------

BURN_IN = 500

ITERATIONS = 2000

THIN = 50


# ------------------------------------------------------------
# Number of top terms used for coherence and interpretation
# ------------------------------------------------------------

TOP_N = 10


# ------------------------------------------------------------
# One candidate model per worker
# ------------------------------------------------------------

MAX_WORKERS = 4


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
    TRAIN_FILE,
    TEST_FILE,
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
# CREATE R CANDIDATE WORKER
# ============================================================

R_WORKER_SCRIPT = (
    CANDIDATE_DIR
    / "evaluate_single_k_scopus.R"
)


R_WORKER_CODE = r'''
args <- commandArgs(
    trailingOnly = TRUE
)

processed_file <- args[1]
train_file     <- args[2]
test_file      <- args[3]

summary_file   <- args[4]
topic_file     <- args[5]
top_terms_file <- args[6]

k              <- as.integer(args[7])
min_doc_freq   <- as.integer(args[8])

random_seed    <- as.integer(args[9])
burn_in        <- as.integer(args[10])
iterations     <- as.integer(args[11])
thin           <- as.integer(args[12])

top_n          <- as.integer(args[13])


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


if (!"processed_text" %in% names(data)) {

    stop(
        "processed_text column not found."
    )
}


processed <- data$processed_text

processed[
    is.na(processed)
] <- ""


# ============================================================
# CONSTRUCT DTM
# ============================================================

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


# ============================================================
# APPLY FINAL DOCUMENT-FREQUENCY FILTER
# ============================================================

document_frequency <- slam::col_sums(
    dtm > 0
)


dtm <- dtm[
    ,
    document_frequency >= min_doc_freq
]


# ============================================================
# LOAD FIXED TRAIN / HELD-OUT SPLIT
# ============================================================

train_id <- read.csv(
    train_file
)$train_id


test_id <- read.csv(
    test_file
)$test_id


dtm_train <- dtm[
    train_id,
]


dtm_test <- dtm[
    test_id,
]


# ============================================================
# RESTRICT VOCABULARY TO TRAINING DATA
# ============================================================

training_term_mask <- (
    slam::col_sums(
        dtm_train
    ) > 0
)


dtm_train <- dtm_train[
    ,
    training_term_mask
]


dtm_test <- dtm_test[
    ,
    training_term_mask
]


# ============================================================
# REMOVE EMPTY DOCUMENTS
# ============================================================

train_nonempty <- (
    slam::row_sums(
        dtm_train
    ) > 0
)


test_nonempty <- (
    slam::row_sums(
        dtm_test
    ) > 0
)


n_empty_train <- sum(
    !train_nonempty
)


n_empty_test <- sum(
    !test_nonempty
)


dtm_train <- dtm_train[
    train_nonempty,
]


dtm_test <- dtm_test[
    test_nonempty,
]


# ============================================================
# DIAGNOSTICS
# ============================================================

cat(
    "K:",
    k,
    "\n"
)


cat(
    "Training documents:",
    nrow(dtm_train),
    "\n"
)


cat(
    "Held-out documents:",
    nrow(dtm_test),
    "\n"
)


cat(
    "Vocabulary:",
    ncol(dtm_train),
    "\n"
)


cat(
    "Minimum DF:",
    min_doc_freq,
    "\n"
)


cat(
    "Top terms:",
    top_n,
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
    "\n\n"
)


flush.console()


# ============================================================
# FIT CANDIDATE LDA MODEL
# ============================================================

cat(
    "Fitting candidate K =",
    k,
    "...\n"
)


flush.console()


start_time <- Sys.time()


model <- topicmodels::LDA(
    dtm_train,
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
# HELD-OUT PERPLEXITY
# ============================================================

heldout_perplexity <- (
    topicmodels::perplexity(
        model,
        newdata = dtm_test
    )
)


# ============================================================
# TOPIC-TERM DISTRIBUTION
# ============================================================

beta <- posterior(
    model
)$terms


# ============================================================
# BINARY TRAINING DTM
#
# Convert the sparse DTM to a sparse binary triplet matrix.
# ============================================================

binary_train <- dtm_train

binary_train$v[
    binary_train$v > 0
] <- 1


term_document_frequency <- (
    slam::col_sums(
        binary_train
    )
)


# ============================================================
# TERM CO-OCCURRENCE MATRIX
#
# Vocabulary is ~1,842 terms, so a dense vocabulary x
# vocabulary co-occurrence matrix is manageable.
#
# Cross-product is computed once, avoiding repeated sparse
# column multiplication during coherence calculation.
# ============================================================

binary_matrix <- as.matrix(
    binary_train
)


storage.mode(
    binary_matrix
) <- "double"


cooccurrence_matrix <- crossprod(
    binary_matrix
)


rm(
    binary_matrix
)


gc()


# ============================================================
# SEMANTIC COHERENCE FUNCTION
#
# Mimno-style coherence:
#
# sum_{m=2}^M sum_{l=1}^{m-1}
# log((D(v_m,v_l)+1) / D(v_l))
# ============================================================

calculate_coherence <- function(
    top_indices,
    cooccurrence_matrix,
    term_df
) {

    score <- 0


    if (length(top_indices) < 2) {

        return(
            score
        )
    }


    for (
        m in 2:length(top_indices)
    ) {

        for (
            l in 1:(m - 1)
        ) {

            word_m <- top_indices[m]

            word_l <- top_indices[l]


            cooccurrence <- (
                cooccurrence_matrix[
                    word_m,
                    word_l
                ]
            )


            denominator <- (
                term_df[
                    word_l
                ]
            )


            if (
                denominator > 0
            ) {

                score <- (
                    score
                    +
                    log(
                        (
                            cooccurrence
                            + 1
                        )
                        /
                        denominator
                    )
                )
            }
        }
    }


    return(
        score
    )
}


# ============================================================
# TOPIC COHERENCE + TOP TERMS
# ============================================================

coherence_values <- numeric(
    k
)


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


    coherence_values[
        topic
    ] <- calculate_coherence(
        top_indices,
        cooccurrence_matrix,
        term_document_frequency
    )


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


# ============================================================
# TOPIC-LEVEL DIAGNOSTICS
# ============================================================

topic_results <- data.frame(

    K = rep(
        k,
        k
    ),

    Topic = seq_len(
        k
    ),

    Coherence = coherence_values
)


# ============================================================
# PAIRWISE COSINE SIMILARITY
#
# Since each beta row sums to 1, cosine similarity between
# all topic vectors can be calculated efficiently by matrix
# multiplication after row normalization.
# ============================================================

beta_norm <- sqrt(
    rowSums(
        beta^2
    )
)


beta_normalized <- (
    beta
    /
    beta_norm
)


similarity_matrix <- (
    beta_normalized
    %*%
    t(
        beta_normalized
    )
)


if (
    k > 1
) {

    similarity_values <- (
        similarity_matrix[
            upper.tri(
                similarity_matrix
            )
        ]
    )

} else {

    similarity_values <- NA_real_
}


# ============================================================
# ELAPSED TIME
# ============================================================

elapsed <- as.numeric(
    difftime(
        Sys.time(),
        start_time,
        units = "secs"
    )
)


# ============================================================
# MODEL-LEVEL SUMMARY
# ============================================================

summary_results <- data.frame(

    K = k,

    Perplexity =
        heldout_perplexity,

    Mean_Coherence =
        mean(
            coherence_values
        ),

    Median_Coherence =
        median(
            coherence_values
        ),

    Min_Coherence =
        min(
            coherence_values
        ),

    Max_Coherence =
        max(
            coherence_values
        ),

    SD_Coherence =
        sd(
            coherence_values
        ),

    Mean_Similarity =
        mean(
            similarity_values,
            na.rm = TRUE
        ),

    Median_Similarity =
        median(
            similarity_values,
            na.rm = TRUE
        ),

    Max_Similarity =
        max(
            similarity_values,
            na.rm = TRUE
        ),

    SD_Similarity =
        sd(
            similarity_values,
            na.rm = TRUE
        ),

    Elapsed_seconds =
        elapsed,

    Training_documents =
        nrow(dtm_train),

    Heldout_documents =
        nrow(dtm_test),

    Vocabulary =
        ncol(dtm_train),

    Empty_training_removed =
        n_empty_train,

    Empty_heldout_removed =
        n_empty_test
)


# ============================================================
# SAVE INDIVIDUAL-K RESULTS
# ============================================================

write.csv(
    summary_results,
    summary_file,
    row.names = FALSE
)


write.csv(
    topic_results,
    topic_file,
    row.names = FALSE
)


write.csv(
    top_term_results,
    top_terms_file,
    row.names = FALSE
)


# ============================================================
# FINAL MESSAGE
# ============================================================

cat(
    sprintf(
        paste0(
            "DONE K=%d",
            " | perplexity=%.4f",
            " | mean coherence=%.4f",
            " | mean similarity=%.4f",
            " | max similarity=%.4f",
            " | %.2f s\n"
        ),
        k,
        heldout_perplexity,
        mean(
            coherence_values
        ),
        mean(
            similarity_values,
            na.rm = TRUE
        ),
        max(
            similarity_values,
            na.rm = TRUE
        ),
        elapsed
    )
)


flush.console()
'''


R_WORKER_SCRIPT.write_text(
    R_WORKER_CODE,
    encoding="utf-8",
)


print()

print(
    "Created R worker:",
    R_WORKER_SCRIPT
)


# ============================================================
# SINGLE-K PYTHON WORKER
# ============================================================

def evaluate_k(k):

    # --------------------------------------------------------
    # Individual output files
    # --------------------------------------------------------

    summary_file = (
        CANDIDATE_DIR
        / f"k_{k:03d}_summary.csv"
    )


    topic_file = (
        CANDIDATE_DIR
        / f"k_{k:03d}_topics.csv"
    )


    top_terms_file = (
        CANDIDATE_DIR
        / f"k_{k:03d}_top_terms.csv"
    )


    log_file = (
        CANDIDATE_DIR
        / f"k_{k:03d}.log"
    )


    # --------------------------------------------------------
    # Resume support
    #
    # Require all three outputs to exist and be readable.
    # --------------------------------------------------------

    if (
        summary_file.exists()
        and
        topic_file.exists()
        and
        top_terms_file.exists()
    ):

        try:

            previous_summary = (
                pd.read_csv(
                    summary_file
                )
            )


            previous_topics = (
                pd.read_csv(
                    topic_file
                )
            )


            previous_terms = (
                pd.read_csv(
                    top_terms_file
                )
            )


            valid_existing = (
                len(
                    previous_summary
                ) == 1
                and
                "K"
                in previous_summary.columns
                and
                int(
                    previous_summary.loc[
                        0,
                        "K"
                    ]
                ) == k
                and
                len(
                    previous_topics
                ) == k
                and
                len(
                    previous_terms
                ) == (
                    k
                    * TOP_N
                )
            )


            if valid_existing:

                return {
                    "K": k,
                    "status":
                        "existing",
                    "summary":
                        summary_file,
                    "topics":
                        topic_file,
                    "terms":
                        top_terms_file,
                    "log":
                        log_file,
                }


        except Exception:

            pass


    # --------------------------------------------------------
    # Command
    # --------------------------------------------------------

    command = [
        str(RSCRIPT),
        str(R_WORKER_SCRIPT),
        str(PROCESSED_FILE),
        str(TRAIN_FILE),
        str(TEST_FILE),
        str(summary_file),
        str(topic_file),
        str(top_terms_file),
        str(k),
        str(MIN_DOC_FREQ),
        str(RANDOM_SEED),
        str(BURN_IN),
        str(ITERATIONS),
        str(THIN),
        str(TOP_N),
    ]


    # --------------------------------------------------------
    # Avoid nested parallelism
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    wall_start = (
        time.time()
    )


    with log_file.open(
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


    # --------------------------------------------------------
    # Validate execution
    # --------------------------------------------------------

    if process.returncode != 0:

        return {
            "K": k,
            "status":
                "failed",
            "returncode":
                process.returncode,
            "wall_seconds":
                wall_elapsed,
            "log":
                log_file,
        }


    required_outputs = [
        summary_file,
        topic_file,
        top_terms_file,
    ]


    if not all(
        path.exists()
        for path
        in required_outputs
    ):

        return {
            "K": k,
            "status":
                "failed",
            "returncode":
                process.returncode,
            "wall_seconds":
                wall_elapsed,
            "log":
                log_file,
        }


    return {
        "K": k,
        "status":
            "completed",
        "wall_seconds":
            wall_elapsed,
        "summary":
            summary_file,
        "topics":
            topic_file,
        "terms":
            top_terms_file,
        "log":
            log_file,
    }


# ============================================================
# COMBINE AVAILABLE RESULTS
# ============================================================

def combine_results():

    summary_frames = []

    topic_frames = []

    term_frames = []


    for k in K_VALUES:

        summary_file = (
            CANDIDATE_DIR
            / f"k_{k:03d}_summary.csv"
        )


        topic_file = (
            CANDIDATE_DIR
            / f"k_{k:03d}_topics.csv"
        )


        terms_file = (
            CANDIDATE_DIR
            / f"k_{k:03d}_top_terms.csv"
        )


        # ----------------------------------------------------
        # Summary
        # ----------------------------------------------------

        if summary_file.exists():

            try:

                summary_frames.append(
                    pd.read_csv(
                        summary_file
                    )
                )

            except Exception as exc:

                print(
                    f"Warning: could not read "
                    f"{summary_file}: {exc}",
                    flush=True,
                )


        # ----------------------------------------------------
        # Topic diagnostics
        # ----------------------------------------------------

        if topic_file.exists():

            try:

                topic_frames.append(
                    pd.read_csv(
                        topic_file
                    )
                )

            except Exception as exc:

                print(
                    f"Warning: could not read "
                    f"{topic_file}: {exc}",
                    flush=True,
                )


        # ----------------------------------------------------
        # Top terms
        # ----------------------------------------------------

        if terms_file.exists():

            try:

                term_frames.append(
                    pd.read_csv(
                        terms_file
                    )
                )

            except Exception as exc:

                print(
                    f"Warning: could not read "
                    f"{terms_file}: {exc}",
                    flush=True,
                )


    # --------------------------------------------------------
    # Combine model summaries
    # --------------------------------------------------------

    combined_summary = None


    if summary_frames:

        combined_summary = (
            pd.concat(
                summary_frames,
                ignore_index=True,
            )
            .sort_values(
                "K"
            )
            .drop_duplicates(
                subset=[
                    "K"
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )


        combined_summary.to_csv(
            SUMMARY_FILE,
            index=False,
        )


    # --------------------------------------------------------
    # Combine topic diagnostics
    # --------------------------------------------------------

    combined_topics = None


    if topic_frames:

        combined_topics = (
            pd.concat(
                topic_frames,
                ignore_index=True,
            )
            .sort_values(
                [
                    "K",
                    "Topic",
                ]
            )
            .drop_duplicates(
                subset=[
                    "K",
                    "Topic",
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )


        combined_topics.to_csv(
            TOPIC_FILE,
            index=False,
        )


    # --------------------------------------------------------
    # Combine top terms
    # --------------------------------------------------------

    combined_terms = None


    if term_frames:

        combined_terms = (
            pd.concat(
                term_frames,
                ignore_index=True,
            )
            .sort_values(
                [
                    "K",
                    "Topic",
                    "Rank",
                ]
            )
            .drop_duplicates(
                subset=[
                    "K",
                    "Topic",
                    "Rank",
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )


        combined_terms.to_csv(
            TOP_TERMS_FILE,
            index=False,
        )


    return (
        combined_summary,
        combined_topics,
        combined_terms,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 76,
        flush=True,
    )


    print(
        "SCOPUS ACADEMIC PARALLEL CANDIDATE MODEL EVALUATION",
        flush=True,
    )


    print(
        "=" * 76,
        flush=True,
    )


    print(
        "Project directory :",
        PROJECT_DIR,
        flush=True,
    )


    print(
        "Rscript           :",
        RSCRIPT,
        flush=True,
    )


    print(
        "Processed corpus  :",
        PROCESSED_FILE,
        flush=True,
    )


    print(
        "Training indices  :",
        TRAIN_FILE,
        flush=True,
    )


    print(
        "Held-out indices  :",
        TEST_FILE,
        flush=True,
    )


    print(
        "Minimum DF        :",
        MIN_DOC_FREQ,
        flush=True,
    )


    print(
        "K values          :",
        K_VALUES,
        flush=True,
    )


    print(
        "Parallel workers  :",
        MAX_WORKERS,
        flush=True,
    )


    print(
        "Top terms         :",
        TOP_N,
        flush=True,
    )


    print(
        "Random seed       :",
        RANDOM_SEED,
        flush=True,
    )


    print(
        "Burn-in           :",
        BURN_IN,
        flush=True,
    )


    print(
        "Iterations        :",
        ITERATIONS,
        flush=True,
    )


    print(
        "Thin              :",
        THIN,
        flush=True,
    )


    print(
        "=" * 76,
        flush=True,
    )


    overall_start = (
        time.time()
    )


    failed_k = []


    # ========================================================
    # PARALLEL EXECUTION
    # ========================================================

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:


        future_to_k = {
            executor.submit(
                evaluate_k,
                k,
            ): k
            for k in K_VALUES
        }


        for future in as_completed(
            future_to_k
        ):


            k = future_to_k[
                future
            ]


            try:

                result = (
                    future.result()
                )


            except Exception as exc:

                print(
                    f"[FAILED] K={k} "
                    f"| Python exception: "
                    f"{exc}",
                    flush=True,
                )


                failed_k.append(
                    k
                )


                continue


            status = (
                result[
                    "status"
                ]
            )


            # ------------------------------------------------
            # Existing result
            # ------------------------------------------------

            if status == "existing":

                summary = pd.read_csv(
                    result[
                        "summary"
                    ]
                )


                print(
                    f"[SKIP] K={k:>2} "
                    f"| already completed "
                    f"| perplexity="
                    f"{summary.loc[0, 'Perplexity']:.4f} "
                    f"| coherence="
                    f"{summary.loc[0, 'Mean_Coherence']:.4f}",
                    flush=True,
                )


            # ------------------------------------------------
            # Newly completed
            # ------------------------------------------------

            elif status == "completed":

                summary = pd.read_csv(
                    result[
                        "summary"
                    ]
                )


                print(
                    f"[DONE] K={k:>2} "
                    f"| perplexity="
                    f"{summary.loc[0, 'Perplexity']:.4f} "
                    f"| coherence="
                    f"{summary.loc[0, 'Mean_Coherence']:.4f} "
                    f"| mean similarity="
                    f"{summary.loc[0, 'Mean_Similarity']:.4f} "
                    f"| max similarity="
                    f"{summary.loc[0, 'Max_Similarity']:.4f} "
                    f"| R time="
                    f"{summary.loc[0, 'Elapsed_seconds']:.2f} s",
                    flush=True,
                )


            # ------------------------------------------------
            # Failed
            # ------------------------------------------------

            else:

                print(
                    f"[FAILED] K={k:>2} "
                    f"| inspect: "
                    f"{result['log']}",
                    flush=True,
                )


                failed_k.append(
                    k
                )


            # ------------------------------------------------
            # Update combined outputs after every result
            # ------------------------------------------------

            combine_results()


    # ========================================================
    # FINAL COMBINATION
    # ========================================================

    (
        combined_summary,
        combined_topics,
        combined_terms,
    ) = combine_results()


    overall_elapsed = (
        time.time()
        - overall_start
    )


    print()

    print(
        "=" * 76,
        flush=True,
    )


    print(
        "FINAL CANDIDATE MODEL RESULTS",
        flush=True,
    )


    print(
        "=" * 76,
        flush=True,
    )


    if combined_summary is not None:

        print()


        display_columns = [
            "K",
            "Perplexity",
            "Mean_Coherence",
            "Median_Coherence",
            "Min_Coherence",
            "Mean_Similarity",
            "Median_Similarity",
            "Max_Similarity",
            "Elapsed_seconds",
        ]


        print(
            combined_summary[
                display_columns
            ]
            .to_string(
                index=False
            ),
            flush=True,
        )


    print()


    if combined_topics is not None:

        print(
            "Topic diagnostic rows:",
            len(
                combined_topics
            ),
            flush=True,
        )


    if combined_terms is not None:

        print(
            "Top-term rows:",
            len(
                combined_terms
            ),
            flush=True,
        )


    print()


    print(
        "Total wall time:",
        f"{overall_elapsed / 60:.2f} minutes",
        flush=True,
    )


    print()


    print(
        "Summary output:",
        SUMMARY_FILE,
        flush=True,
    )


    print(
        "Topic output  :",
        TOPIC_FILE,
        flush=True,
    )


    print(
        "Top terms     :",
        TOP_TERMS_FILE,
        flush=True,
    )


    print(
        "Individual files:",
        CANDIDATE_DIR,
        flush=True,
    )


    # ========================================================
    # FAILURE STATUS
    # ========================================================

    if failed_k:

        print()


        print(
            "Failed K values:",
            failed_k,
            flush=True,
        )


        print(
            "Inspect the corresponding "
            "k_XXX.log files.",
            flush=True,
        )


        sys.exit(1)


    print()


    print(
        "All candidate models "
        "completed successfully.",
        flush=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()