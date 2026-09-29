
args <- commandArgs(
    trailingOnly = TRUE
)

processed_file <- args[1]
dtm_file       <- args[2]
vocab_file     <- args[3]
min_doc_freq   <- as.integer(args[4])

suppressPackageStartupMessages({
    library(tm)
    library(slam)
})

MIN_TERM_LENGTH <- 3

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
# Verify no empty documents
# ------------------------------------------------------------

empty_documents <- (
    slam::row_sums(dtm) == 0
)

if (any(empty_documents)) {

    stop(
        paste(
            "Final DTM contains",
            sum(empty_documents),
            "empty documents."
        )
    )
}

# ------------------------------------------------------------
# Convert to sparse triplet representation
# ------------------------------------------------------------

dtm_triplet <- as.simple_triplet_matrix(
    dtm
)

triplet_output <- data.frame(
    Document_ID = dtm_triplet$i,
    Term_ID = dtm_triplet$j,
    Frequency = dtm_triplet$v
)

# ------------------------------------------------------------
# Vocabulary
# ------------------------------------------------------------

vocabulary_output <- data.frame(
    Term_ID = seq_len(
        ncol(dtm)
    ),
    Term = Terms(dtm),
    stringsAsFactors = FALSE
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

write.csv(
    triplet_output,
    dtm_file,
    row.names = FALSE
)

write.csv(
    vocabulary_output,
    vocab_file,
    row.names = FALSE
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
    sum(dtm),
    "\n"
)

cat(
    "Nonzero entries:",
    length(dtm_triplet$v),
    "\n"
)

cat(
    "Empty documents:",
    sum(empty_documents),
    "\n"
)
