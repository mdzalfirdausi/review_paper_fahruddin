
args <- commandArgs(
    trailingOnly = TRUE
)

processed_file <- args[1]
summary_file   <- args[2]
terms_file     <- args[3]
min_doc_freq   <- as.integer(args[4])

suppressPackageStartupMessages({
    library(tm)
    library(slam)
})

MIN_TERM_LENGTH <- 3
MIN_DOC_FREQ <- min_doc_freq

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

valid <- (
    !is.na(processed)
    & nzchar(trimws(processed))
)

processed <- processed[valid]

# ------------------------------------------------------------
# Construct raw DTM
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Apply document-frequency filtering
# ------------------------------------------------------------

document_frequency <- slam::col_sums(
    dtm > 0
)

keep_terms <- (
    document_frequency
    >= MIN_DOC_FREQ
)

dtm <- dtm[
    ,
    keep_terms
]

# ------------------------------------------------------------
# Check empty documents
# ------------------------------------------------------------

document_tokens <- slam::row_sums(
    dtm
)

empty_documents <- (
    document_tokens == 0
)

n_empty <- sum(
    empty_documents
)

# ------------------------------------------------------------
# Final frequencies
# ------------------------------------------------------------

final_document_frequency <- (
    slam::col_sums(
        dtm > 0
    )
)

final_term_frequency <- (
    slam::col_sums(
        dtm
    )
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

summary <- data.frame(
    Documents = dtm$nrow,
    Vocabulary = dtm$ncol,
    Tokens = sum(dtm$v),
    Minimum_DF = MIN_DOC_FREQ,
    Min_Term_Length = MIN_TERM_LENGTH,
    Empty_Documents = n_empty
)

terms <- data.frame(
    Term = dtm$dimnames$Terms,
    Document_Frequency = as.numeric(
        final_document_frequency
    ),
    Term_Frequency = as.numeric(
        final_term_frequency
    ),
    stringsAsFactors = FALSE
)

terms <- terms[
    order(
        -terms$Document_Frequency,
        -terms$Term_Frequency
    ),
]

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

write.csv(
    summary,
    summary_file,
    row.names = FALSE
)

write.csv(
    terms,
    terms_file,
    row.names = FALSE
)

# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

cat(
    "Documents:",
    dtm$nrow,
    "\n"
)

cat(
    "Vocabulary:",
    dtm$ncol,
    "\n"
)

cat(
    "Tokens:",
    sum(dtm$v),
    "\n"
)

cat(
    "Minimum DF:",
    MIN_DOC_FREQ,
    "\n"
)

cat(
    "Empty documents:",
    n_empty,
    "\n"
)
