
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
    5, 10, 15, 20, 25, 30, 40, 50, 60
)

RANDOM_SEED <- 123

# Lightweight Gibbs configuration for coarse search.
BURN_IN <- 200
ITERATIONS <- 800
THIN <- 20

# ------------------------------------------------------------
# Load processed corpus
# ------------------------------------------------------------

data <- read.csv(
    processed_file,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    fileEncoding = "UTF-8"
)

processed <- data$processed_text
processed[is.na(processed)] <- ""

# ------------------------------------------------------------
# Construct DTM
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
# Apply final document-frequency threshold
# ------------------------------------------------------------

document_frequency <- slam::col_sums(
    dtm > 0
)

dtm <- dtm[
    ,
    document_frequency >= min_doc_freq
]

# ------------------------------------------------------------
# Load fixed train / held-out split
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Restrict vocabulary to terms occurring in training data
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
# Remove empty documents, if any
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

# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

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
    "\n\n"
)

# ------------------------------------------------------------
# Coarse K search
# ------------------------------------------------------------

results <- data.frame()

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

    elapsed <- as.numeric(
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
                elapsed
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
            elapsed
        )
    )

    flush.console()
}
