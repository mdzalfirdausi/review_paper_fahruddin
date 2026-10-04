
args <- commandArgs(
    trailingOnly = TRUE
)

processed_file <- args[1]
train_file     <- args[2]
test_file      <- args[3]
summary_file   <- args[4]
topic_file     <- args[5]
top_terms_file <- args[6]
min_doc_freq   <- as.integer(args[7])

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
    20,
    30,
    40,
    50
)

BURN_IN <- 500
ITERATIONS <- 2000
THIN <- 50

RANDOM_SEED <- 123

TOP_N <- 10

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

if (any(is.na(processed))) {
    stop(
        "Processed corpus contains missing text."
    )
}

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
# Apply final document-frequency filter
# ------------------------------------------------------------

document_frequency <- slam::col_sums(
    dtm > 0
)

dtm <- dtm[
    ,
    document_frequency >= min_doc_freq
]

# ------------------------------------------------------------
# Fixed train/test split
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
# Restrict vocabulary to training data
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
# Remove empty documents
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
# Binary training DTM
# ------------------------------------------------------------

binary_train <- dtm_train

binary_train$v[] <- 1

# ------------------------------------------------------------
# Term document frequencies
# ------------------------------------------------------------

term_document_frequency <- (
    slam::col_sums(
        binary_train
    )
)

# ------------------------------------------------------------
# Term co-occurrence matrix
#
# C[i,j] = number of training documents in which
# terms i and j occur together.
# ------------------------------------------------------------

term_cooccurrence <- (
    slam::crossprod_simple_triplet_matrix(
        binary_train
    )
)

# ------------------------------------------------------------
# Semantic coherence
#
# Mimno-style coherence:
#
# sum_{m=2}^{M}
# sum_{l=1}^{m-1}
# log(
#     (D(v_m, v_l) + 1)
#     / D(v_l)
# )
# ------------------------------------------------------------

calculate_coherence <- function(
    top_indices,
    cooccurrence,
    term_df
) {

    score <- 0

    if (length(top_indices) < 2) {
        return(score)
    }

    for (m in 2:length(top_indices)) {

        for (l in 1:(m - 1)) {

            word_m <- top_indices[m]
            word_l <- top_indices[l]

            cooccur <- (
                cooccurrence[
                    word_m,
                    word_l
                ]
            )

            denominator <- (
                term_df[
                    word_l
                ]
            )

            if (denominator > 0) {

                score <- score + log(
                    (
                        cooccur + 1
                    )
                    /
                    denominator
                )
            }
        }
    }

    return(score)
}

# ------------------------------------------------------------
# Cosine similarity
# ------------------------------------------------------------

cosine_similarity <- function(
    x,
    y
) {

    denominator <- (
        sqrt(
            sum(
                x^2
            )
        )
        *
        sqrt(
            sum(
                y^2
            )
        )
    )

    if (denominator == 0) {
        return(0)
    }

    return(
        sum(
            x * y
        )
        /
        denominator
    )
}

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
    n_empty_train,
    "\n"
)

cat(
    "Empty held-out removed:",
    n_empty_test,
    "\n\n"
)

# ------------------------------------------------------------
# Output containers
# ------------------------------------------------------------

summary_results <- data.frame()

topic_results <- data.frame()

top_term_results <- data.frame()

# ------------------------------------------------------------
# Candidate models
# ------------------------------------------------------------

for (k in K_VALUES) {

    cat(
        "Fitting K =",
        k,
        "... "
    )

    flush.console()

    start_time <- Sys.time()

    # --------------------------------------------------------
    # Fit LDA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Held-out perplexity
    # --------------------------------------------------------

    heldout_perplexity <- (
        topicmodels::perplexity(
            model,
            newdata = dtm_test
        )
    )

    # --------------------------------------------------------
    # Topic-term probabilities
    # --------------------------------------------------------

    beta <- posterior(
        model
    )$terms

    # --------------------------------------------------------
    # Topic coherence
    # --------------------------------------------------------

    coherence_values <- numeric(
        k
    )

    for (topic in seq_len(k)) {

        top_indices <- order(
            beta[
                topic,
            ],
            decreasing = TRUE
        )[
            seq_len(
                min(
                    TOP_N,
                    ncol(beta)
                )
            )
        ]

        coherence_values[
            topic
        ] <- calculate_coherence(
            top_indices,
            term_cooccurrence,
            term_document_frequency
        )

        # ----------------------------------------------------
        # Save top terms
        # ----------------------------------------------------

        top_words <- colnames(
            beta
        )[
            top_indices
        ]

        for (
            rank in seq_along(
                top_words
            )
        ) {

            top_term_results <- rbind(
                top_term_results,
                data.frame(
                    K = k,
                    Topic = topic,
                    Rank = rank,
                    Term = top_words[
                        rank
                    ],
                    Beta = beta[
                        topic,
                        top_indices[
                            rank
                        ]
                    ]
                )
            )
        }
    }

    # --------------------------------------------------------
    # Pairwise topic similarity
    # --------------------------------------------------------

    similarity_values <- c()

    if (k > 1) {

        for (
            i in 1:(k - 1)
        ) {

            for (
                j in (i + 1):k
            ) {

                similarity_values <- c(
                    similarity_values,
                    cosine_similarity(
                        beta[
                            i,
                        ],
                        beta[
                            j,
                        ]
                    )
                )
            }
        }
    }

    # --------------------------------------------------------
    # Topic-level results
    # --------------------------------------------------------

    topic_results <- rbind(
        topic_results,
        data.frame(
            K = k,
            Topic = seq_len(k),
            Coherence =
                coherence_values
        )
    )

    # --------------------------------------------------------
    # Elapsed time
    # --------------------------------------------------------

    elapsed <- as.numeric(
        difftime(
            Sys.time(),
            start_time,
            units = "secs"
        )
    )

    # --------------------------------------------------------
    # Model-level summary
    # --------------------------------------------------------

    summary_results <- rbind(
        summary_results,
        data.frame(
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

            Mean_Similarity =
                mean(
                    similarity_values
                ),

            Median_Similarity =
                median(
                    similarity_values
                ),

            Max_Similarity =
                max(
                    similarity_values
                ),

            Elapsed_seconds =
                elapsed
        )
    )

    # --------------------------------------------------------
    # Save incrementally
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    cat(
        sprintf(
            paste0(
                "perplexity = %.4f",
                " | mean coherence = %.4f",
                " | mean similarity = %.4f",
                " | %.2f s\n"
            ),
            heldout_perplexity,
            mean(
                coherence_values
            ),
            mean(
                similarity_values
            ),
            elapsed
        )
    )

    flush.console()
}
