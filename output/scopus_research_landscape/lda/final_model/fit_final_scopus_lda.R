
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
