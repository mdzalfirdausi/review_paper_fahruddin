#!/usr/bin/env python3

# ============================================================
# Parallel coarse LDA K-search for the full Scopus corpus
#
# Designed for:
#   issnode1
#   Intel Core i9-14900K
#   32 logical CPUs
#   188 GiB RAM
#
# Each candidate K is fitted in an independent R process.
# ============================================================

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import os
import subprocess
import sys
import time

import pandas as pd


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

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

PARALLEL_DIR = (
    LDA_DIR
    / "coarse_parallel"
)

PARALLEL_DIR.mkdir(
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
# CHECK REQUIRED R PACKAGES
# ============================================================

print("Checking R environment...")
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
print(
    RSCRIPT
)

print()

print("Required R packages:")
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
# OUTPUT FILES
# ============================================================

FINAL_OUTPUT_FILE = (
    LDA_DIR
    / "scopus_academic_coarse_k_search.csv"
)

MASTER_LOG_FILE = (
    LDA_DIR
    / "scopus_academic_coarse_parallel.log"
)


# ============================================================
# LDA CONFIGURATION
# ============================================================

K_VALUES = [
    5,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
]


# Final DF threshold from notebook:
#
# 0.5% * 16,409 documents
# = ceil(82.045)
# = 83
#
MIN_DOC_FREQ = 83


RANDOM_SEED = 123


# ------------------------------------------------------------
# Lightweight Gibbs configuration for coarse search
# ------------------------------------------------------------

BURN_IN = 200

ITERATIONS = 800

THIN = 20


# ------------------------------------------------------------
# Parallel workers
#
# There are only 9 candidate K values, so there is no benefit
# in using more than 9 workers.
#
# Start with 8 to avoid unnecessary CPU/cache contention.
# ------------------------------------------------------------

MAX_WORKERS = 8


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

print()
print("Checking input files...")
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
# CREATE SINGLE-K R WORKER
# ============================================================

R_WORKER_SCRIPT = (
    PARALLEL_DIR
    / "fit_single_k_scopus.R"
)


R_WORKER_CODE = r'''
args <- commandArgs(
    trailingOnly = TRUE
)

processed_file <- args[1]
train_file     <- args[2]
test_file      <- args[3]
output_file    <- args[4]

k              <- as.integer(args[5])
min_doc_freq   <- as.integer(args[6])

random_seed    <- as.integer(args[7])
burn_in        <- as.integer(args[8])
iterations     <- as.integer(args[9])
thin           <- as.integer(args[10])


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
    "Empty training removed:",
    n_empty_train,
    "\n"
)


cat(
    "Empty held-out removed:",
    n_empty_test,
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
# FIT LDA MODEL
# ============================================================

cat(
    "Fitting K =",
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
# SAVE RESULT
# ============================================================

result <- data.frame(

    K = k,

    Perplexity =
        heldout_perplexity,

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


write.csv(
    result,
    output_file,
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
            " | elapsed=%.2f s\n"
        ),
        k,
        heldout_perplexity,
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

def fit_k(k):

    output_file = (
        PARALLEL_DIR
        / f"k_{k:03d}.csv"
    )


    log_file = (
        PARALLEL_DIR
        / f"k_{k:03d}.log"
    )


    # --------------------------------------------------------
    # Resume support
    #
    # If a valid result already exists, do not recompute it.
    # --------------------------------------------------------

    if output_file.exists():

        try:

            previous = pd.read_csv(
                output_file
            )


            if (
                len(previous) == 1
                and
                "K" in previous.columns
                and
                int(
                    previous.loc[
                        0,
                        "K"
                    ]
                ) == k
                and
                "Perplexity"
                in previous.columns
            ):

                return {
                    "K": k,
                    "status": "existing",
                    "file": output_file,
                    "log": log_file,
                }


        except Exception:

            pass


    # --------------------------------------------------------
    # R command
    # --------------------------------------------------------

    command = [
        str(RSCRIPT),
        str(R_WORKER_SCRIPT),
        str(PROCESSED_FILE),
        str(TRAIN_FILE),
        str(TEST_FILE),
        str(output_file),
        str(k),
        str(MIN_DOC_FREQ),
        str(RANDOM_SEED),
        str(BURN_IN),
        str(ITERATIONS),
        str(THIN),
    ]


    # --------------------------------------------------------
    # Prevent nested BLAS/OpenMP oversubscription.
    #
    # We parallelize across K values, so each R worker should
    # remain effectively single-threaded internally.
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
    # Run
    # --------------------------------------------------------

    wall_start = time.time()


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
    # Check result
    # --------------------------------------------------------

    if process.returncode != 0:

        return {
            "K": k,
            "status": "failed",
            "returncode":
                process.returncode,
            "wall_seconds":
                wall_elapsed,
            "log":
                log_file,
        }


    if not output_file.exists():

        return {
            "K": k,
            "status": "failed",
            "returncode":
                process.returncode,
            "wall_seconds":
                wall_elapsed,
            "log":
                log_file,
        }


    return {
        "K": k,
        "status": "completed",
        "wall_seconds":
            wall_elapsed,
        "file":
            output_file,
        "log":
            log_file,
    }


# ============================================================
# COMBINE AVAILABLE RESULTS
# ============================================================

def combine_results():

    result_frames = []


    for k in K_VALUES:

        result_file = (
            PARALLEL_DIR
            / f"k_{k:03d}.csv"
        )


        if not result_file.exists():

            continue


        try:

            result_df = pd.read_csv(
                result_file
            )


            if len(
                result_df
            ) == 0:

                continue


            result_frames.append(
                result_df
            )


        except Exception as exc:

            print(
                f"Warning: could not read "
                f"{result_file}: {exc}",
                flush=True,
            )


    if not result_frames:

        return None


    combined = (
        pd.concat(
            result_frames,
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


    combined.to_csv(
        FINAL_OUTPUT_FILE,
        index=False,
    )


    return combined


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 72,
        flush=True,
    )

    print(
        "SCOPUS ACADEMIC PARALLEL COARSE K SEARCH",
        flush=True,
    )

    print(
        "=" * 72,
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
        "=" * 72,
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
                fit_k,
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
                    f"| Python exception: {exc}",
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

                result_df = pd.read_csv(
                    result[
                        "file"
                    ]
                )


                perplexity = float(
                    result_df.loc[
                        0,
                        "Perplexity"
                    ]
                )


                print(
                    f"[SKIP] K={k:>2} "
                    f"| already completed "
                    f"| perplexity="
                    f"{perplexity:.4f}",
                    flush=True,
                )


            # ------------------------------------------------
            # Newly completed
            # ------------------------------------------------

            elif status == "completed":

                result_df = pd.read_csv(
                    result[
                        "file"
                    ]
                )


                perplexity = float(
                    result_df.loc[
                        0,
                        "Perplexity"
                    ]
                )


                elapsed = float(
                    result_df.loc[
                        0,
                        "Elapsed_seconds"
                    ]
                )


                print(
                    f"[DONE] K={k:>2} "
                    f"| perplexity="
                    f"{perplexity:.4f} "
                    f"| R time="
                    f"{elapsed:.2f} s",
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
            # Update combined output after every result
            # ------------------------------------------------

            combine_results()


    # ========================================================
    # FINAL COMBINATION
    # ========================================================

    combined = (
        combine_results()
    )


    overall_elapsed = (
        time.time()
        - overall_start
    )


    print()
    print(
        "=" * 72,
        flush=True,
    )

    print(
        "FINAL COARSE K RESULTS",
        flush=True,
    )

    print(
        "=" * 72,
        flush=True,
    )


    if combined is not None:

        print()

        print(
            combined[
                [
                    "K",
                    "Perplexity",
                    "Elapsed_seconds",
                ]
            ]
            .to_string(
                index=False
            ),
            flush=True,
        )


    print()

    print(
        "Total wall time:",
        f"{overall_elapsed / 60:.2f} minutes",
        flush=True,
    )


    print(
        "Combined result:",
        FINAL_OUTPUT_FILE,
        flush=True,
    )


    print(
        "Individual results:",
        PARALLEL_DIR,
        flush=True,
    )


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
        "All candidate K values "
        "completed successfully.",
        flush=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()