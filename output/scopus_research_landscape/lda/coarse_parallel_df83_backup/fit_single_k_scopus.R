
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
