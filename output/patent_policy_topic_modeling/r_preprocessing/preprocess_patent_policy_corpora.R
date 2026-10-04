
args <- commandArgs(
    trailingOnly = TRUE
)

input_file  <- args[1]
output_file <- args[2]

suppressPackageStartupMessages({
    library(tm)
    library(SnowballC)
})

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

query_stopwords <- c(
    "power",
    "flow",
    "machine",
    "learning",
    "optimization",
    "optimisation"
)

# ------------------------------------------------------------
# Load corpus
# ------------------------------------------------------------

data <- read.csv(
    input_file,
    stringsAsFactors = FALSE,
    check.names = FALSE,
    fileEncoding = "UTF-8"
)

if (!"document_id" %in% names(data)) {
    stop(
        "Input file does not contain document_id."
    )
}

if (!"Abstract" %in% names(data)) {
    stop(
        "Input file does not contain Abstract."
    )
}

abstracts <- data$Abstract

# ------------------------------------------------------------
# Construct corpus
# ------------------------------------------------------------

corpus <- VCorpus(
    VectorSource(
        abstracts
    )
)

# ------------------------------------------------------------
# Text preprocessing
# ------------------------------------------------------------

corpus <- tm_map(
    corpus,
    content_transformer(tolower)
)

corpus <- tm_map(
    corpus,
    removePunctuation
)

corpus <- tm_map(
    corpus,
    removeNumbers
)

# Explicitly remove ©
corpus <- tm_map(
    corpus,
    content_transformer(
        function(x) {
            gsub(
                "\u00A9",
                " ",
                x,
                fixed = TRUE
            )
        }
    )
)

corpus <- tm_map(
    corpus,
    stripWhitespace
)

corpus <- tm_map(
    corpus,
    removeWords,
    stopwords("english")
)

corpus <- tm_map(
    corpus,
    removeWords,
    query_stopwords
)

corpus <- tm_map(
    corpus,
    stemDocument,
    language = "english"
)

corpus <- tm_map(
    corpus,
    stripWhitespace
)

# ------------------------------------------------------------
# Extract processed text
# ------------------------------------------------------------

processed_text <- vapply(
    corpus,
    as.character,
    character(1)
)

output <- data.frame(
    document_id = data$document_id,
    processed_text = processed_text,
    stringsAsFactors = FALSE
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

write.csv(
    output,
    output_file,
    row.names = FALSE,
    fileEncoding = "UTF-8"
)

# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

cat(
    "Documents processed:",
    nrow(output),
    "\n"
)

cat(
    "Empty processed documents:",
    sum(
        !nzchar(
            trimws(
                output$processed_text
            )
        )
    ),
    "\n"
)
