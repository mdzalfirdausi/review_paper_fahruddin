
args <- commandArgs(trailingOnly = TRUE)


# ============================================================
# Arguments
# ============================================================

processed_file       <- args[1]
metadata_file        <- args[2]
summary_file         <- args[3]
top_terms_file       <- args[4]
theta_file           <- args[5]
document_topics_file <- args[6]
prevalence_file      <- args[7]
representative_file  <- args[8]

final_k         <- as.integer(args[9])
min_doc_freq    <- as.integer(args[10])
random_seed     <- as.integer(args[11])
burn_in         <- as.integer(args[12])
iterations      <- as.integer(args[13])
thin            <- as.integer(args[14])
top_n           <- as.integer(args[15])
representative_n <- as.integer(args[16])


# ============================================================
# Packages
# ============================================================

suppressPackageStartupMessages({
    library(tm)
    library(slam)
    library(topicmodels)
})


# ============================================================
# Configuration
# ============================================================

MIN_TERM_LENGTH <- 3

set.seed(random_seed)


# ============================================================
# Validate input files
# ============================================================

if (!file.exists(processed_file)) {
    stop(
        paste(
            "Processed corpus not found:",
            processed_file
        )
    )
}


if (!file.exists(metadata_file)) {
    stop(
        paste(
            "Metadata file not found:",
            metadata_file
        )
    )
}


# ============================================================
# Read processed corpus and metadata
# ============================================================

data <- read.csv(
    processed_file,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    fileEncoding = "UTF-8"
)


metadata <- read.csv(
    metadata_file,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    fileEncoding = "UTF-8"
)


# ============================================================
# Validate required processed-text column
# ============================================================

if (!("processed_text" %in% colnames(data))) {
    stop(
        paste(
            "processed_text column not found in:",
            processed_file
        )
    )
}


# ============================================================
# Validate initial row alignment
# ============================================================

if (nrow(data) != nrow(metadata)) {
    stop(
        paste(
            "Processed corpus and metadata have different",
            "numbers of rows:",
            nrow(data),
            "versus",
            nrow(metadata)
        )
    )
}


cat(
    "Processed rows:",
    nrow(data),
    "\n"
)

cat(
    "Metadata rows:",
    nrow(metadata),
    "\n"
)


# ============================================================
# Identify valid processed documents
# ============================================================

processed <- data[["processed_text"]]


valid <- (
    !is.na(processed)
    &
    nzchar(trimws(processed))
)


n_invalid <- sum(!valid)


cat(
    "Empty processed documents before DTM:",
    n_invalid,
    "\n"
)


# ------------------------------------------------------------
# Preserve original row number BEFORE filtering
# ------------------------------------------------------------

original_row <- seq_len(nrow(metadata))


# ------------------------------------------------------------
# Apply the same validity mask everywhere
# ------------------------------------------------------------

data <- data[
    valid,
    ,
    drop = FALSE
]


metadata <- metadata[
    valid,
    ,
    drop = FALSE
]


processed <- processed[valid]


metadata$Original_Row <- original_row[valid]


cat(
    "Documents before final DTM:",
    length(processed),
    "\n"
)


# ============================================================
# Construct raw full-corpus DTM
# ============================================================

corpus <- VCorpus(
    VectorSource(processed)
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
    "Raw vocabulary:",
    ncol(dtm),
    "\n"
)


cat(
    "Raw tokens:",
    sum(dtm$v),
    "\n"
)


# ============================================================
# Apply minimum document-frequency filter
# ============================================================

document_frequency <- slam::col_sums(
    dtm > 0
)


keep_terms <- (
    document_frequency
    >=
    min_doc_freq
)


n_removed_terms <- sum(
    !keep_terms
)


dtm <- dtm[
    ,
    keep_terms
]


cat(
    "Minimum document frequency:",
    min_doc_freq,
    "\n"
)


cat(
    "Terms removed by DF filter:",
    n_removed_terms,
    "\n"
)


cat(
    "Vocabulary after DF filtering:",
    ncol(dtm),
    "\n"
)


# ============================================================
# Identify documents that become empty after DF filtering
# ============================================================

document_tokens <- slam::row_sums(
    dtm
)


nonempty <- (
    document_tokens > 0
)


n_empty <- sum(
    !nonempty
)


if (n_empty > 0) {

    dtm <- dtm[
        nonempty,
        ,
        drop = FALSE
    ]

    data <- data[
        nonempty,
        ,
        drop = FALSE
    ]

    metadata <- metadata[
        nonempty,
        ,
        drop = FALSE
    ]

}


cat(
    "Empty documents removed after DF filtering:",
    n_empty,
    "\n"
)


cat(
    "Final documents:",
    nrow(dtm),
    "\n"
)


cat(
    "Final vocabulary:",
    ncol(dtm),
    "\n"
)


cat(
    "Final tokens:",
    sum(dtm$v),
    "\n"
)


# ============================================================
# Final DTM validation
# ============================================================

if (nrow(dtm) == 0) {
    stop(
        "Final DTM contains zero documents."
    )
}


if (ncol(dtm) == 0) {
    stop(
        "Final DTM contains zero terms."
    )
}


if (nrow(dtm) != nrow(metadata)) {
    stop(
        paste(
            "DTM/metadata alignment failure:",
            nrow(dtm),
            "DTM rows versus",
            nrow(metadata),
            "metadata rows."
        )
    )
}


# ============================================================
# Fit final full-corpus LDA
# ============================================================

cat("\n")

cat(
    "Fitting final K =",
    final_k,
    "LDA model on full corpus...\n"
)


cat(
    "Seed:",
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


cat("\n")


start_time <- Sys.time()


final_model <- LDA(
    dtm,
    k = final_k,
    method = "Gibbs",
    control = list(
        seed = random_seed,
        burnin = burn_in,
        iter = iterations,
        thin = thin
    )
)


elapsed_seconds <- as.numeric(
    difftime(
        Sys.time(),
        start_time,
        units = "secs"
    )
)


# ============================================================
# Extract posterior distributions
# ============================================================

posterior_values <- posterior(
    final_model
)


theta <- posterior_values$topics

beta <- posterior_values$terms


# ============================================================
# Validate posterior dimensions
# ============================================================

if (nrow(theta) != nrow(dtm)) {
    stop(
        "Theta/document dimension mismatch."
    )
}


if (ncol(theta) != final_k) {
    stop(
        "Theta/topic dimension mismatch."
    )
}


if (nrow(beta) != final_k) {
    stop(
        "Beta/topic dimension mismatch."
    )
}


if (ncol(beta) != ncol(dtm)) {
    stop(
        "Beta/vocabulary dimension mismatch."
    )
}


# ============================================================
# Probability validation
# ============================================================

theta_error <- max(
    abs(
        rowSums(theta)
        - 1
    )
)


beta_error <- max(
    abs(
        rowSums(beta)
        - 1
    )
)


cat(
    "Theta max row-sum error:",
    theta_error,
    "\n"
)


cat(
    "Beta max row-sum error:",
    beta_error,
    "\n"
)


# ============================================================
# Topic prevalence
# ============================================================

topic_prevalence <- colMeans(
    theta
)


topic_prevalence <- (
    topic_prevalence
    /
    sum(topic_prevalence)
)


prevalence <- data.frame(
    Topic = seq_len(final_k),

    Prevalence = as.numeric(
        topic_prevalence
    ),

    Prevalence_Percent = (
        100
        *
        as.numeric(topic_prevalence)
    ),

    stringsAsFactors = FALSE
)


prevalence <- prevalence[
    order(
        -prevalence$Prevalence
    ),
    ,
    drop = FALSE
]


prevalence$Prevalence_Rank <- seq_len(
    nrow(prevalence)
)


prevalence <- prevalence[
    ,
    c(
        "Topic",
        "Prevalence_Rank",
        "Prevalence",
        "Prevalence_Percent"
    ),
    drop = FALSE
]


# ============================================================
# Extract top terms
# ============================================================

top_term_rows <- list()

row_index <- 1


for (topic_index in seq_len(final_k)) {

    term_order <- order(
        beta[
            topic_index,
        ],
        decreasing = TRUE
    )


    n_selected <- min(
        top_n,
        length(term_order)
    )


    selected <- term_order[
        seq_len(n_selected)
    ]


    for (rank_index in seq_along(selected)) {

        term_index <- selected[
            rank_index
        ]


        top_term_rows[[
            row_index
        ]] <- data.frame(
            Topic = topic_index,

            Rank = rank_index,

            Term = colnames(beta)[
                term_index
            ],

            Beta = beta[
                topic_index,
                term_index
            ],

            stringsAsFactors = FALSE
        )


        row_index <- row_index + 1

    }

}


top_terms <- do.call(
    rbind,
    top_term_rows
)


# ============================================================
# Complete document-topic probability matrix
# ============================================================

theta_output <- data.frame(
    Document_ID = seq_len(
        nrow(theta)
    ),

    Original_Row = metadata$Original_Row,

    theta,

    check.names = FALSE
)


colnames(theta_output)[
    -(1:2)
] <- paste0(
    "Topic_",
    seq_len(final_k)
)


# ============================================================
# Dominant topic for every document
# ============================================================

dominant_topic <- apply(
    theta,
    1,
    which.max
)


dominant_probability <- apply(
    theta,
    1,
    max
)


document_topics <- data.frame(
    Document_ID = seq_len(
        nrow(theta)
    ),

    Original_Row = metadata$Original_Row,

    Topic = dominant_topic,

    Topic_Probability = dominant_probability,

    stringsAsFactors = FALSE
)


# ------------------------------------------------------------
# Attach metadata
# ------------------------------------------------------------

metadata_columns <- setdiff(
    colnames(metadata),
    "Original_Row"
)


document_topics <- cbind(
    document_topics,
    metadata[
        ,
        metadata_columns,
        drop = FALSE
    ]
)


# ============================================================
# Representative documents
# ============================================================

representative_rows <- list()

row_index <- 1


for (topic_index in seq_len(final_k)) {

    topic_probabilities <- theta[
        ,
        topic_index
    ]


    document_order <- order(
        topic_probabilities,
        decreasing = TRUE
    )


    n_selected <- min(
        representative_n,
        length(document_order)
    )


    selected_documents <- document_order[
        seq_len(n_selected)
    ]


    for (
        rank_index
        in seq_along(selected_documents)
    ) {

        document_index <- selected_documents[
            rank_index
        ]


        representative_row <- data.frame(
            Topic = topic_index,

            Rank = rank_index,

            Document_ID = document_index,

            Original_Row = metadata$Original_Row[
                document_index
            ],

            Topic_Probability = theta[
                document_index,
                topic_index
            ],

            stringsAsFactors = FALSE
        )


        representative_row <- cbind(
            representative_row,

            metadata[
                document_index,
                metadata_columns,
                drop = FALSE
            ]
        )


        representative_rows[[
            row_index
        ]] <- representative_row


        row_index <- row_index + 1

    }

}


representative_documents <- do.call(
    rbind,
    representative_rows
)


# ============================================================
# Model summary
# ============================================================

model_summary <- data.frame(
    K = final_k,

    Documents = nrow(dtm),

    Vocabulary = ncol(dtm),

    Tokens = sum(dtm$v),

    Minimum_DF = min_doc_freq,

    Min_Term_Length = MIN_TERM_LENGTH,

    Empty_Processed_Documents = n_invalid,

    Empty_Documents_Removed_After_DF = n_empty,

    Random_Seed = random_seed,

    Burn_In = burn_in,

    Iterations = iterations,

    Thin = thin,

    Top_Terms = top_n,

    Representative_Documents = representative_n,

    Theta_Max_Row_Sum_Error = theta_error,

    Beta_Max_Row_Sum_Error = beta_error,

    Topic_Prevalence_Sum = sum(
        topic_prevalence
    ),

    Elapsed_Seconds = elapsed_seconds,

    stringsAsFactors = FALSE
)


# ============================================================
# Save outputs
# ============================================================

write.csv(
    model_summary,
    summary_file,
    row.names = FALSE
)


write.csv(
    top_terms,
    top_terms_file,
    row.names = FALSE
)


write.csv(
    theta_output,
    theta_file,
    row.names = FALSE
)


write.csv(
    document_topics,
    document_topics_file,
    row.names = FALSE
)


write.csv(
    prevalence,
    prevalence_file,
    row.names = FALSE
)


write.csv(
    representative_documents,
    representative_file,
    row.names = FALSE
)


# ============================================================
# Final diagnostics
# ============================================================

cat("\n")

cat(
    "============================================================\n"
)


cat(
    "FINAL LDA COMPLETE\n"
)


cat(
    "============================================================\n"
)


cat(
    "K:",
    final_k,
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
    "Tokens:",
    sum(dtm$v),
    "\n"
)


cat(
    "Minimum DF:",
    min_doc_freq,
    "\n"
)


cat(
    "Minimum term length:",
    MIN_TERM_LENGTH,
    "\n"
)


cat(
    "Theta max row-sum error:",
    theta_error,
    "\n"
)


cat(
    "Beta max row-sum error:",
    beta_error,
    "\n"
)


cat(
    "Topic prevalence sum:",
    sum(topic_prevalence),
    "\n"
)


cat(
    "Elapsed seconds:",
    elapsed_seconds,
    "\n"
)


cat(
    "============================================================\n"
)
