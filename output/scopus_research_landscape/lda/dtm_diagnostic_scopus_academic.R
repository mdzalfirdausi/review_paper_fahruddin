
args <- commandArgs(
    trailingOnly = TRUE
)

processed_file <- args[1]
summary_file   <- args[2]
terms_file     <- args[3]

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

if (!"processed_text" %in% names(data)) {
    stop(
        "processed_text column not found."
    )
}

processed <- data$processed_text

processed[is.na(processed)] <- ""

# ------------------------------------------------------------
# Construct corpus
# ------------------------------------------------------------

corpus <- VCorpus(
    VectorSource(
        processed
    )
)

# ------------------------------------------------------------
# Construct raw DTM
# ------------------------------------------------------------

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
# Diagnostics
# ------------------------------------------------------------

document_totals <- slam::row_sums(
    dtm
)

term_totals <- slam::col_sums(
    dtm
)

document_frequency <- slam::col_sums(
    dtm > 0
)

summary_output <- data.frame(
    Documents = nrow(dtm),
    Vocabulary = ncol(dtm),
    Tokens = sum(dtm),
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
        document_frequency
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
    "Empty documents:",
    sum(document_totals == 0),
    "\n"
)
