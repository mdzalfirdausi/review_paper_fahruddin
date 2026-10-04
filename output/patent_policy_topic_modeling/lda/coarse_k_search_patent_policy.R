
args <- commandArgs(
    trailingOnly = TRUE
)

processed_file <- args[1]
train_file     <- args[2]
test_file      <- args[3]
output_file    <- args[4]
min_doc_freq   <- as.integer(args[5])

suppressPackageStartupMessages({
    library(tm)
    library(slam)
    library(topicmodels)
})

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MIN_TERM_LENGTH <- 3

K_VALUES <- c(
    5,
    10,
    15,
    20,
    25,
    30,
    40
)

BURN_IN <- 50
ITERATIONS <- 100
THIN <- 10
RANDOM_SEED <- 123

# ------------------------------------------------------------
# Load processed documents
# ------------------------------------------------------------

data <- read.csv(
    processed_file,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    fileEncoding = "UTF-8"
)

processed <- data$processed_text

if (any(is.na(processed))) {
    stop(
        "Processed corpus contains missing text."
    )
}

# ------------------------------------------------------------
# Reconstruct raw DTM
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Apply final corpus-level DF filter
# ------------------------------------------------------------

document_frequency <- slam::col_sums(
    dtm > 0
)

keep_terms <- (
    document_frequency
    >= min_doc_freq
)

dtm <- dtm[
    ,
    keep_terms
]

# ------------------------------------------------------------
# Load fixed train/test indices
# ------------------------------------------------------------

train_id <- read.csv(
    train_file
)$train_id

test_id <- read.csv(
    test_file
)$test_id

# Validate split.
if (
    length(
        intersect(
            train_id,
            test_id
        )
    ) > 0
) {
    stop(
        "Training and test indices overlap."
    )
}

if (
    length(
        union(
            train_id,
            test_id
        )
    ) != dtm$nrow
) {
    stop(
        "Training/test indices do not cover all documents."
    )
}

# ------------------------------------------------------------
# Split DTM
# ------------------------------------------------------------

dtm_train <- dtm[
    train_id,
]

dtm_test <- dtm[
    test_id,
]

# ------------------------------------------------------------
# Restrict to terms occurring in training data
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Remove empty documents if necessary
# ------------------------------------------------------------

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

empty_train <- sum(
    !train_nonempty
)

empty_test <- sum(
    !test_nonempty
)

dtm_train <- dtm_train[
    train_nonempty,
]

dtm_test <- dtm_test[
    test_nonempty,
]

# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

cat(
    "Training documents:",
    dtm_train$nrow,
    "\n"
)

cat(
    "Held-out documents:",
    dtm_test$nrow,
    "\n"
)

cat(
    "Vocabulary:",
    dtm_train$ncol,
    "\n"
)

cat(
    "Minimum DF:",
    min_doc_freq,
    "\n"
)

cat(
    "Empty training removed:",
    empty_train,
    "\n"
)

cat(
    "Empty held-out removed:",
    empty_test,
    "\n\n"
)

# ------------------------------------------------------------
# Candidate K search
# ------------------------------------------------------------

results <- data.frame(
    K = integer(),
    Perplexity = numeric(),
    Elapsed_seconds = numeric()
)

for (k in K_VALUES) {

    cat(
        "Fitting K =",
        k,
        "... "
    )

    flush.console()

    start_time <- Sys.time()

    model <- topicmodels::LDA(
        dtm_train,
        k = k,
        method = "Gibbs",
        control = list(
            seed = RANDOM_SEED,
            burnin = BURN_IN,
            iter = ITERATIONS,
            thin = THIN
        )
    )

    heldout_perplexity <- (
        topicmodels::perplexity(
            model,
            newdata = dtm_test
        )
    )

    elapsed_seconds <- as.numeric(
        difftime(
            Sys.time(),
            start_time,
            units = "secs"
        )
    )

    results <- rbind(
        results,
        data.frame(
            K = k,
            Perplexity =
                heldout_perplexity,
            Elapsed_seconds =
                elapsed_seconds
        )
    )

    # Save incrementally.
    write.csv(
        results,
        output_file,
        row.names = FALSE
    )

    cat(
        sprintf(
            "perplexity = %.4f | %.2f s\n",
            heldout_perplexity,
            elapsed_seconds
        )
    )

    flush.console()
}
