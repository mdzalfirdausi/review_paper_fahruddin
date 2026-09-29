
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

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

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
# Document-frequency filtering
# ------------------------------------------------------------

document_frequency <- slam::col_sums(
    dtm > 0
)

dtm <- dtm[
    ,
    document_frequency >= min_doc_freq
]

# ------------------------------------------------------------
# Final diagnostics
# ------------------------------------------------------------

document_totals <- slam::row_sums(
    dtm
)

term_totals <- slam::col_sums(
    dtm
)

final_document_frequency <- slam::col_sums(
    dtm > 0
)

summary_output <- data.frame(
    Documents = nrow(dtm),
    Vocabulary = ncol(dtm),
    Tokens = sum(dtm),
    Minimum_DF = min_doc_freq,
    Min_Term_Length = MIN_TERM_LENGTH,
    Empty_Documents = sum(
        document_totals == 0
    )
)

term_output <- data.frame(
    Term = names(term_totals),
    Frequency = as.numeric(
        term_totals
    ),
    Document_Frequency = as.numeric(
        final_document_frequency
    ),
    stringsAsFactors = FALSE
)

term_output <- term_output[
    order(
        -term_output$Frequency,
        term_output$Term
    ),
]

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

write.csv(
    summary_output,
    summary_file,
    row.names = FALSE
)

write.csv(
    term_output,
    terms_file,
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
    "Minimum DF:",
    min_doc_freq,
    "\n"
)

cat(
    "Empty documents:",
    sum(document_totals == 0),
    "\n"
)
