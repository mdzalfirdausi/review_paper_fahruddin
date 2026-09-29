
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
    "energy", "power", "electricity", "generation", "demand", "load", "consumption", "forecast", "forecasts", "forecasted", "forecasting", "predict", "predicts", "predicted", "predicting", "prediction", "predictions", "predictive"
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

if (!"Text" %in% names(data)) {
    stop(
        "Input file does not contain a Text column."
    )
}

text <- data$Text

text[is.na(text)] <- ""

# ------------------------------------------------------------
# Construct corpus
# ------------------------------------------------------------

corpus <- VCorpus(
    VectorSource(
        text
    )
)

# ------------------------------------------------------------
# Text preprocessing
# ------------------------------------------------------------

corpus <- tm_map(
    corpus,
    content_transformer(
        tolower
    )
)

corpus <- tm_map(
    corpus,
    removeNumbers
)

corpus <- tm_map(
    corpus,
    removePunctuation
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

processed_text <- sapply(
    corpus,
    content
)

processed_text <- trimws(
    processed_text
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output <- data.frame(
    Document_ID = data$Document_ID,
    processed_text = processed_text,
    stringsAsFactors = FALSE
)

write.csv(
    output,
    output_file,
    row.names = FALSE,
    fileEncoding = "UTF-8"
)

cat(
    "Documents processed:",
    nrow(output),
    "\n"
)

cat(
    "Empty documents:",
    sum(
        nchar(
            output$processed_text
        ) == 0
    ),
    "\n"
)
