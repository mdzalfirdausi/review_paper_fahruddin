
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
