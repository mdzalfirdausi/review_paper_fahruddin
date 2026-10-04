
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
# Frequencies
# ------------------------------------------------------------

document_frequency <- (
    slam::col_sums(
        dtm > 0
    )
)

term_frequency <- (
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
    Min_Term_Length = MIN_TERM_LENGTH
)

# ------------------------------------------------------------
# Term diagnostics
# ------------------------------------------------------------

terms <- data.frame(
    Term = dtm$dimnames$Terms,
    Document_Frequency = as.numeric(
        document_frequency
    ),
    Term_Frequency = as.numeric(
        term_frequency
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

cat(
    "Documents:",
    dtm$nrow,
    "\n"
)

cat(
    "Raw vocabulary:",
    dtm$ncol,
    "\n"
)

cat(
    "Tokens:",
    sum(dtm$v),
    "\n"
)
