# Forecasting-related research landscape

This repository analyzes forecasting-related scholarly publications from Scopus and the Scopus publications cited by patents (Lens) or policy documents (Overton). The workflows match publications by DOI, build three text corpora, fit separate latent Dirichlet allocation (LDA) models, interpret their topics, and compare academic and external-citation themes. Patent and policy exports supply citation metadata; the matched Scopus records supply the scholarly titles and abstracts used for LDA.

> **Rerun notice:** The checked-in notebooks and scripts were developed with earlier topic-number searches and some still contain ML–OPF wording, fixed `K=50`/`K=20` topic labels, and old macro-theme mappings. For a new search over **every integer K from 2 through 25**, follow [Rerunning LDA for K 2 to 25](#rerunning-lda-for-k-2-to-25) before interpreting or publishing any results. Changing the candidate K values does **not** automatically change the final model.

## Requirements and installation

- Git, Python with Conda (or a compatible environment manager), R, and JupyterLab.
- Python packages used by these files: `pandas`, `numpy`, `scipy`, `matplotlib`, `matplotlib-venn`, and `openpyxl` (for `.xlsx` files).
- R packages used by the generated R backends: `tm`, `SnowballC`, `slam`, and `topicmodels`.
- `tmux` is optional for keeping long commands attached to a server session. On a shared cluster, use the computing resources and job scheduler according to the site's rules.

### Windows setup

Windows is supported as a setup target; Ubuntu is not required. Install Git for Windows, a Python/Conda distribution, and [R for Windows](https://cran.r-project.org/bin/windows/base/). RStudio is optional: this workflow calls `Rscript.exe` from Python. Install R packages from the R console:

```r
install.packages(c("tm", "SnowballC", "slam", "topicmodels"), repos = "https://cloud.r-project.org")
sapply(c("tm", "SnowballC", "slam", "topicmodels"), requireNamespace, quietly = TRUE)
```

All four checks should return `TRUE`. Windows binary packages generally avoid compilation; if an installation requires compiling source packages, use the matching [Rtools](https://cran.r-project.org/bin/windows/Rtools/) release. See the [R Windows FAQ](https://cran.r-project.org/bin/windows/base/rw-FAQ.html).

In a Conda-enabled PowerShell terminal, create the Python environment:

```powershell
conda create -n forecasting-lda -c conda-forge python=3.11 pandas numpy scipy matplotlib matplotlib-venn openpyxl jupyterlab
conda activate forecasting-lda
python --version
git --version
```

If PowerShell does not recognize Conda, initialize PowerShell from your Conda installation prompt with `conda init powershell`, then reopen the terminal. An existing environment can also be used if all required packages are installed.

Clone the repository only if you do not already have a checkout:

```powershell
Set-Location 'M:\projects_latex'
git clone https://github.com/mdzalfirdausi/review_paper_fahruddin.git
Set-Location 'M:\projects_latex\review_paper_fahruddin'
jupyter lab
```

If you already have the project, activate the environment and go directly to its directory. Configure GitHub authentication first if the repository requires it.

### Configure Rscript for your operating system

The supplied modeling files contain the ISS-specific path `/nfs/mfirdausi/miniconda3/envs/pytorch/bin/Rscript`. Replace `RSCRIPT` in **all four `scripts/5.scopus_*_iss.py` scripts** and the setup cells of the academic and patent/policy modeling notebooks. For Windows, use the actual installed executable, for example:

```python
from pathlib import Path

# Replace R-X.Y.Z with your installed version/directory.
RSCRIPT = Path(r"C:\Program Files\R\R-X.Y.Z\bin\Rscript.exe")
assert RSCRIPT.is_file(), f"Rscript not found: {RSCRIPT}"
```

Alternatively, if Rscript is already on the terminal/kernel's PATH, use this on either operating system:

```python
from pathlib import Path
from shutil import which

rscript_executable = which("Rscript")
if rscript_executable is None:
    raise FileNotFoundError("Add Rscript to PATH or configure its absolute path.")
RSCRIPT = Path(rscript_executable)
```

Restart Jupyter after changing PATH. `Get-Command Rscript` checks discovery in PowerShell; `which Rscript` does the same in Bash. The `_iss` filenames identify the supplied scripts; keep their filenames and adapt configuration for Windows. Their Windows execution has not been tested here.

### Optional Linux / ISS setup

On ISS, use the installed Conda environment or create one with Python and R:

```bash
conda create -n forecasting-lda -c conda-forge python=3.11 pandas numpy scipy matplotlib matplotlib-venn openpyxl jupyterlab r-base r-tm r-snowballc r-slam r-topicmodels
conda activate forecasting-lda
which python
which Rscript
Rscript -e 'sapply(c("tm","SnowballC","slam","topicmodels"), requireNamespace, quietly=TRUE)'
cd ~/project/review_paper_fahruddin
jupyter lab
```

Git and optional tmux must be installed on that Linux host. Follow cluster instructions for modules, package installation, and compute allocation. Native Windows PowerShell does not use the Bash/tmux commands shown later; those are for an SSH session on ISS or a separately configured Linux/WSL environment.

## Expected layout and data

Run the Python scripts from the repository root. They determine `PROJECT_DIR` as the parent of `scripts/`. Open notebooks with the repository root as the working directory or from `notebooks/`; most modeling notebooks detect either location. **The citation-landscape notebook is an exception:** its `PROJECT_DIR = Path.cwd().resolve().parent` assumes the notebook kernel starts in `notebooks/`. Set `PROJECT_DIR` explicitly if starting it at the root.

For Windows notebooks, including notebooks 6 and 7, replace the project-root assignment in the setup cell with your checkout path:

```python
from pathlib import Path
PROJECT_DIR = Path(r"M:\projects_latex\review_paper_fahruddin")
```

On ISS, the corresponding assignment is:

```python
PROJECT_DIR = Path.home() / "project" / "review_paper_fahruddin"
```

The required Scopus filename is **`scopus_full.xlsx`**. Its Windows location is **`M:\projects_latex\review_paper_fahruddin\data\scopus\scopus_full.xlsx`**; the repository-relative path remains `data/scopus/scopus_full.xlsx` on either system. Keep the standalone scripts in `scripts/` so their existing project-root detection works.

```text
review_paper_fahruddin/
├── scripts/
│   ├── 0.doi_filter_overton.py
│   ├── extract_lens_dois.py
│   ├── extract_patent_cited_scholar.py
│   ├── extract_overton_cited_scholar.py
│   └── 5.scopus_*_iss.py
├── notebooks/
│   ├── 3.topic_modeling_patent_policy.ipynb
│   ├── 4.topic_analysis_patent_policy.ipynb
│   ├── 5.scopus_research_landscape_iss.ipynb
│   ├── 6.patent_policy_citation_landscape.ipynb
│   └── 7.academic_patent_policy_theme_comparison.ipynb
├── data/
│   ├── scopus/scopus_full.xlsx
│   ├── lens/<Lens patent-citation export>.xlsx
│   ├── overton/overton_raw.xlsx
│   └── lda/                 # DOI-matched scholarly corpora
└── output/                  # generated results
```

The attached copies of some notebooks have numbered suffixes such as `(3)` or `(5)`; the descriptions below refer to those supplied versions. If you use a different GitHub revision, check its cell headings and configuration values before running it.

The input exports are not made by these notebooks. Obtain the Scopus, Lens, and Overton files legitimately and place them at the paths configured in each script. In particular, `extract_lens_dois.py` and `extract_patent_cited_scholar.py` contain a specific dated Lens export filename; change their `INPUT_FILE` / `LENS_EXPORT_FILE` values when using a new export. The Overton matching script expects `data/overton/overton_raw.xlsx`. Keep DOI matching and document order consistent across the processed corpus, metadata, and train/test index files.

## Local Scopus requirements and Git exclusion

The repository does **not** include the Scopus export. Supply your own export locally at **`data/scopus/scopus_full.xlsx`** before running the workflow. The scripts read the first Excel worksheet by default. Export the forecasting-related search results with citation information, bibliographic information, abstracts, keywords, affiliations, and funding fields; preserve these exact headers for the complete matching and analysis workflow:

```text
Link
Authors
Author full names
Author(s) ID
Title
Year
Source title
Cited by
Affiliations
Publisher
Abbreviated Source Title
DOI
Abstract
Author Keywords
Index Keywords
Funding Details
Funding Texts
```

The academic corpus preparation requires `Title`, `Year`, `DOI`, and `Abstract`; the larger column set above is needed by the matching scripts and downstream citation analyses. Keep genuinely missing values empty. Academic modeling combines titles and abstracts and excludes records without usable text; patent/policy modeling uses the matched abstracts. DOI matching requires usable DOI values, so an unmatched publication is not automatically evidence of zero external citations.

Create the local input directories from the repository root:

```powershell
New-Item -ItemType Directory -Force -Path data/scopus,data/lens,data/overton,data/lda
```

Or in Linux Bash:

```bash
mkdir -p data/scopus data/lens data/overton data/lda
```

Add this rule to the repository-root `.gitignore` to keep the Scopus export local:

```gitignore
# Local Scopus source data
/data/scopus/
```

If Lens/Overton exports and all derived document-level data are also intended to stay local, use the following broader rules instead. Aggregate tables and figures outside these directories can still be selected for publication:

```gitignore
# Local source exports and matched scholarly records
/data/
# Intermediate/model outputs can contain scholarly titles and abstracts
/output/scopus_research_landscape/lda/
/output/patent_policy_topic_modeling/metadata/
/output/patent_policy_topic_modeling/r_preprocessing/
/output/patent_policy_topic_modeling/lda/
.ipynb_checkpoints/
__pycache__/
/logs/
```

Inspect notebook outputs and other exported tables before committing: ignoring the input file does not exclude text copied into notebooks or other outputs. Check the rule with:

```bash
git check-ignore -v data/scopus/scopus_full.xlsx
git status --short
```

If `data/scopus/` was already tracked, `.gitignore` alone will not untrack it. In that case, `git rm -r --cached -- data/scopus` removes it from the Git index while retaining the local files; it does not remove earlier committed copies from repository history.

## Workflow and file roles

| Order | File | Role and principal output |
| --- | --- | --- |
| 1 (optional) | `scripts/0.doi_filter_overton.py` | Cleans DOIs from a configured raw Overton CSV, writes `data/cleaned_dois.txt`, recovery/rejection lists, and Scopus DOI queries. This is a preparation aid when using that CSV export; it does not replace `data/overton/overton_raw.xlsx` required downstream. Run from the repository root. |
| 2 | `scripts/extract_lens_dois.py` | Extracts DOI values from the configured Lens `.xlsx` export into `data/lens/lens_patent_cited_dois.txt`. |
| 3 | `scripts/extract_patent_cited_scholar.py` | Matches Lens DOIs and patent-citation measures to `data/scopus/scopus_full.xlsx`; writes `data/lda/lens_from_scopus_scholar.xlsx`. |
| 4 | `scripts/extract_overton_cited_scholar.py` | Matches Overton DOI records to Scopus scholarly metadata and abstracts; writes `data/lda/overton_from_scopus_scholar.xlsx`. |
| 5 | `notebooks/3.topic_modeling_patent_policy.ipynb` | Separately preprocesses the matched patent/policy abstracts, constructs document-term matrices, performs K searches and candidate evaluation, estimates final LDA models, and saves topic prevalences, terms, representative publications, and interpretations under `output/patent_policy_topic_modeling/`. |
| 6 | `notebooks/4.topic_analysis_patent_policy.ipynb` | Loads the saved patent/policy model-selection results and compares candidate diagnostics, including perplexity plots. Run after notebook 3 has saved its model-selection files. |
| 7 | `notebooks/5.scopus_research_landscape_iss.ipynb` | Prepares the full Scopus corpus; creates the processed academic text, metadata, and train/test split under `output/scopus_research_landscape/lda/`. It also shows publication trends and loads or fits model-selection results. For an ISS parallel rerun, use it to prepare inputs, then run the standalone scripts below. |
| 8 | `scripts/5.scopus_coarse_k_parallel_iss.py` | Runs independent short-chain Scopus fits for the `K_VALUES` list. Saves per-K results under `output/scopus_research_landscape/lda/coarse_parallel/` and a combined `scopus_academic_coarse_k_search.csv`. |
| 9 (optional) | `scripts/5.scopus_extended_k_parallel_iss.py` | Runs additional short-chain K values in `extended_parallel/`. It is **not needed** when the coarse script already covers K=2–25. The academic notebook currently expects its output, so bypass that old extended-results load or adapt it as described below. |
| 10 | `scripts/5.scopus_candidate_evaluation_parallel_iss.py` | Fits the selected `K_VALUES` with longer chains; writes held-out perplexity, coherence, topic similarity, and top terms to `scopus_academic_candidate_{evaluation,topic_diagnostics,top_terms}.csv`. |
| 11 | `scripts/5.scopus_final_lda_iss.py` | Once K is chosen, refits that `FINAL_K` on the full academic corpus and writes final prevalence, terms, document-topic probabilities, and representative documents. |
| 12 | `notebooks/6.patent_policy_citation_landscape.ipynb` | Uses the matched scholarly corpora for citation counts, leading publications, journals, authors, countries, and figures; it is separate from LDA fitting. |
| 13 | `notebooks/7.academic_patent_policy_theme_comparison.ipynb` | Reads all three *final* topic interpretations, maps topics to common macro-themes, compares prevalence and similarities, and saves tables/figures. Run only after new topic labels and maps have been reviewed. |

`notebooks/5.scopus_research_landscape.ipynb` is another academic variant; use **one** academic workflow consistently rather than executing both academic notebooks against the same output directory. The supplied ISS variant has forecasting-related academic label dictionaries, whereas the supplied non-ISS variant still has old ML–OPF topic labels.

### Typical commands for input matching

From the repository root, after placing the exports and checking their configured filenames, run these commands in either a Conda-enabled Windows terminal or Linux Bash:

```bash
python scripts/extract_lens_dois.py
python scripts/extract_patent_cited_scholar.py
python scripts/extract_overton_cited_scholar.py
```

Optionally run `python scripts/0.doi_filter_overton.py` first when working from its configured raw CSV. The patent/policy topic-modeling notebook and citation-landscape notebook consume the two matched `.xlsx` files; the academic notebook consumes `data/scopus/scopus_full.xlsx`.

## Which files perform topic modeling, and where are models saved?

Three independent LDA models are fitted: the full academic Scopus corpus, patent-cited scholarly publications matched from Lens, and policy-cited scholarly publications matched from Overton. **Lens and Overton identify cited scholarly records; the matched Scopus abstracts supply the modeling text. These are not models of patent full text or policy-document full text.**

| Corpus | Modeling input | Preparation and fitting files | Final output directory |
| --- | --- | --- | --- |
| Academic Scopus | `data/scopus/scopus_full.xlsx` (titles and abstracts) | Prepare text and the split in `notebooks/5.scopus_research_landscape_iss.ipynb`; search K with `scripts/5.scopus_coarse_k_parallel_iss.py`; evaluate candidates with `scripts/5.scopus_candidate_evaluation_parallel_iss.py`; fit the selected final model with `scripts/5.scopus_final_lda_iss.py`. The extended-search script is optional. | `output/scopus_research_landscape/lda/` |
| Lens / patent-cited scholarly corpus | `data/lda/lens_from_scopus_scholar.xlsx` (matched Scopus abstracts) | `scripts/extract_lens_dois.py` and `scripts/extract_patent_cited_scholar.py` prepare the matched data. **`notebooks/3.topic_modeling_patent_policy.ipynb` performs preprocessing, candidate evaluation, and final LDA fitting** for the `patent_cited` corpus. | `output/patent_policy_topic_modeling/lda/`, filenames prefixed `patent_cited_` |
| Overton / policy-cited scholarly corpus | `data/lda/overton_from_scopus_scholar.xlsx` (matched Scopus abstracts) | `scripts/extract_overton_cited_scholar.py` prepares the matched data. **The same `notebooks/3.topic_modeling_patent_policy.ipynb` fits a separate LDA model** for the `policy_cited` corpus. | `output/patent_policy_topic_modeling/lda/`, filenames prefixed `policy_cited_` |

`notebooks/4.topic_analysis_patent_policy.ipynb` analyzes existing model-selection results. Notebook 6 performs citation analysis. Notebook 7 compares manually harmonized themes. These three notebooks do not fit new LDA models. The alternative `5.scopus_research_landscape.ipynb` shares the academic output locations; use one academic notebook consistently.

### Academic Scopus final model outputs

| Saved file (relative to repository root) | Contents |
| --- | --- |
| `output/scopus_research_landscape/lda/scopus_academic_final_model_summary.csv` | Final-model summary, including the selected K. |
| `output/scopus_research_landscape/lda/final_model/scopus_academic_final_beta.csv` | Topic–term probabilities (beta). |
| `output/scopus_research_landscape/lda/scopus_academic_final_document_topics.csv` | Document–topic probabilities with document metadata. |
| `output/scopus_research_landscape/lda/scopus_academic_final_top_terms.csv` | Highest-probability terms for each topic. |
| `output/scopus_research_landscape/lda/scopus_academic_final_topic_prevalence.csv` | Probability-weighted topic prevalence. |
| `output/scopus_research_landscape/lda/scopus_academic_final_representative_documents.csv` | Representative scholarly publications for each topic. |
| `output/scopus_research_landscape/lda/scopus_academic_topic_labeling.csv` | Interpretation worksheet, produced/updated by the academic notebook after fitting. |

### Lens and Overton final model outputs

All the following files are under **`output/patent_policy_topic_modeling/lda/`**. Replace `{corpus}` with `patent_cited` for Lens or `policy_cited` for Overton:

| Filename | Contents |
| --- | --- |
| `{corpus}_final_model_summary.csv` | Summary of that corpus's selected final model. |
| `{corpus}_final_theta.csv` | Document–topic probability matrix (theta). |
| `{corpus}_final_document_topics.csv` | Document-level topic results. |
| `{corpus}_final_metadata.csv` | Metadata aligned to the final modeling corpus. |
| `{corpus}_final_top_terms.csv` | Highest-probability terms for each topic. |
| `{corpus}_final_topic_prevalence.csv` | Probability-weighted topic prevalence. |
| `{corpus}_final_representative_documents.csv` | Representative publications for each topic. |
| `{corpus}_final_topic_interpretation.csv` | Reviewed topic labels and interpretation information. |

For example, the Lens probability matrix is `output/patent_policy_topic_modeling/lda/patent_cited_final_theta.csv`; the Overton matrix is `output/patent_policy_topic_modeling/lda/policy_cited_final_theta.csv`.

### Model outputs versus a reusable fitted model

**The supplied code exports model results to CSV; it does not serialize the complete fitted R LDA object with `saveRDS()`.** A directory named `final_model/` does not mean an `.rds` model exists. The Scopus script exports beta and document-topic results; the patent/policy notebook exports theta and top terms, but does not save a complete beta matrix or a serialized model in the supplied version.

Use these CSVs for interpretation, prevalence tables, and figures. If a reusable fitted object is needed for later R analysis or inference on new documents, add `saveRDS()` to the relevant final-fitting R backend and preserve its vocabulary and preprocessing configuration. That would be an additional code change; this README update does not implement model-object serialization. Rerunning the existing final-fitting stages writes the same output filenames, so archive previous outputs if they must be retained.

On Windows, prepend `M:\projects_latex\review_paper_fahruddin\` to the relative paths above. On ISS, prepend `~/project/review_paper_fahruddin/`. For example, Scopus results are in `M:\projects_latex\review_paper_fahruddin\output\scopus_research_landscape\lda\` on the illustrated Windows checkout.

## Exact inputs, processing, and outputs

All paths below are **relative to the repository root** (`M:\projects_latex\review_paper_fahruddin` on Windows, or `~/project/review_paper_fahruddin` on ISS). Filenames reflect the supplied code; dated export names must be changed in the corresponding configuration if your exports differ. Each filename in an output row belongs to the directory shown in that row. `{corpus}` means **both** `patent_cited` and `policy_cited`; `{K:03d}` is a zero-padded topic count, for example `002` or `025`.

These are generated data, diagnostic, table, figure, and log files. The modeling code also writes R backend scripts automatically. Outputs from later notebook cells exist only after those cells have run successfully.

### `scripts/0.doi_filter_overton.py`

**Input**

- `data/overton_policy_doc_2026-09-10_12-24-30.csv`

**Process:** Read the `Cited DOIs` column, clean and deduplicate DOI values, and create Scopus DOI search batches.

**Output**

| Directory | Filename(s) |
| --- | --- |
| `data/` | `cleaned_dois.txt`, `recovered_dois.txt`, `rejected_values.txt`, `scopus_doi_queries.txt` |

This optional input is directly under `data/`, not `data/overton/`. This script does not generate `overton_raw.xlsx`.

### `scripts/extract_lens_dois.py`

**Input**

- `data/lens/lenspatcite-export-M_Muna-84f6bb81-32bc-4ec9-97c6-f970293a9752-2026-09-25_01-48-07_cited.xlsx`

**Process:** Extract unique DOI values from the `citation external id` column.

**Output**

| Directory | Filename(s) |
| --- | --- |
| `data/lens/` | `lens_patent_cited_dois.txt` |

### `scripts/extract_patent_cited_scholar.py`

**Input**

- `data/lens/lens_patent_cited_dois.txt`
- `data/lens/lenspatcite-export-M_Muna-84f6bb81-32bc-4ec9-97c6-f970293a9752-2026-09-25_01-48-07_cited.xlsx`
- `data/scopus/scopus_full.xlsx`

**Process:** Match normalized DOIs to Scopus records and attach Lens patent-citation and citing-family counts. The Lens export requires `citation external id`, `cited by patent count`, and `citing family count`.

**Output**

| Directory | Filename(s) |
| --- | --- |
| `data/lda/` | `lens_from_scopus_scholar.xlsx` |

### `scripts/extract_overton_cited_scholar.py`

**Input**

- `data/overton/overton_raw.xlsx`
- `data/scopus/scopus_full.xlsx`

**Process:** Match Overton DOI records to Scopus metadata and abstracts. Required Overton headers are `Title`, `DOI`, `Journal`, `Published on`, `Policy citation count`, `Type`, `Publisher`, `Authors`, `Your tags`, and `ORCIDs`.

**Output**

| Directory | Filename(s) |
| --- | --- |
| `data/lda/` | `overton_from_scopus_scholar.xlsx` |

### `notebooks/3.topic_modeling_patent_policy.ipynb`

**Input**

- `data/lda/lens_from_scopus_scholar.xlsx`
- `data/lda/overton_from_scopus_scholar.xlsx`

**Process:** Prepare the two abstract corpora independently, preprocess text in R, build document-term matrices, compare candidate K values, fit final LDA models, and manually interpret and aggregate topics.

**Output**

| Directory | Filename(s) |
| --- | --- |
| `output/patent_policy_topic_modeling/metadata/` | `{corpus}_metadata.csv` |
| `output/patent_policy_topic_modeling/r_preprocessing/` | `{corpus}_abstracts.csv`, `{corpus}_processed.csv`, `{corpus}_raw_dtm_summary.csv`, `{corpus}_raw_dtm_terms.csv`, `{corpus}_final_dtm_summary.csv`, `{corpus}_final_dtm_terms.csv` |
| `output/patent_policy_topic_modeling/lda/` | `{corpus}_train_indices.csv`, `{corpus}_test_indices.csv`, `{corpus}_coarse_k_search.csv`, `{corpus}_extended_k_search.csv`, `{corpus}_candidate_evaluation.csv`, `{corpus}_candidate_topic_diagnostics.csv`, `{corpus}_candidate_top_terms.csv`, `{corpus}_final_metadata.csv`, `{corpus}_final_model_summary.csv`, `{corpus}_final_top_terms.csv`, `{corpus}_final_theta.csv`, `{corpus}_final_document_topics.csv`, `{corpus}_final_topic_prevalence.csv`, `{corpus}_final_representative_documents.csv`, `{corpus}_final_topic_interpretation.csv`, `patent_policy_macro_theme_table.csv` |
| `output/patent_policy_topic_modeling/model_selection/` | `{corpus}_train_indices.csv`, `{corpus}_test_indices.csv`, `{corpus}_coarse_k_search.csv`, `{corpus}_extended_k_search.csv`, `{corpus}_candidate_evaluation.csv`, `{corpus}_candidate_topic_diagnostics.csv`, `{corpus}_candidate_top_terms.csv`, `{corpus}_combined_k_search.csv`, `patent_policy_model_selection.xlsx` |
| `output/patent_policy_topic_modeling/figures/` | `patent_policy_macro_theme_difference.pdf`, `patent_policy_macro_theme_difference.png`, `patent_policy_macro_theme_dumbbell.pdf`, `patent_policy_macro_theme_dumbbell.png` |

Final interpretation and prevalence CSVs are written under `lda/`, even though other topic-related directories are created. Manual mappings must be rebuilt after selecting a new final K.

### `notebooks/4.topic_analysis_patent_policy.ipynb`

**Input**

- `output/patent_policy_topic_modeling/model_selection/{corpus}_candidate_evaluation.csv`
- `output/patent_policy_topic_modeling/model_selection/{corpus}_candidate_topic_diagnostics.csv`
- `output/patent_policy_topic_modeling/model_selection/{corpus}_candidate_top_terms.csv`
- `output/patent_policy_topic_modeling/model_selection/{corpus}_combined_k_search.csv`

**Process:** Read saved model diagnostics, inspect candidate topics and perplexity improvements, and plot model-selection curves. This notebook does not fit LDA.

**Output**

| Directory | Filename(s) |
| --- | --- |
| `output/patent_policy_topic_modeling/figures/` | `{corpus}_perplexity_by_k.pdf` |

Other diagnostic tables are displayed in the notebook; they are not additional saved CSV outputs.

### `notebooks/5.scopus_research_landscape_iss.ipynb`

**Input**

- `data/scopus/scopus_full.xlsx`
- `data/lda/lens_from_scopus_scholar.xlsx`
- `data/lda/overton_from_scopus_scholar.xlsx`

**Process:** Prepare academic title/abstract text and publication-growth tables; preprocess the corpus, inspect term frequencies, construct the final DTM, and create the train/test split. Later cells load the standalone model-selection/final outputs listed below and create diagnostics and topic labels.

**Output**

| Directory | Filename(s) |
| --- | --- |
| `output/scopus_research_landscape/tables/` | `scopus_publication_growth.csv` |
| `output/scopus_research_landscape/figures/` | `scopus_publication_growth.pdf`, `academic_growth_patent_policy_annotated.pdf`, `scopus_df_vocabulary_retention.pdf`, `scopus_academic_k_search.pdf`, `scopus_academic_k_marginal_improvement.pdf`, `scopus_academic_candidate_coherence.pdf` |
| `output/scopus_research_landscape/lda/r_preprocessing/` | `scopus_academic_processed.csv` |
| `output/scopus_research_landscape/lda/` | `scopus_academic_corpus.csv`, `scopus_academic_raw_dtm_summary.csv`, `scopus_academic_raw_term_frequency.csv`, `scopus_academic_df_diagnostics.csv`, `scopus_academic_final_dtm_summary.csv`, `scopus_academic_final_dtm_terms.csv`, `scopus_academic_final_dtm.csv`, `scopus_academic_final_vocabulary.csv`, `scopus_academic_train_indices.csv`, `scopus_academic_test_indices.csv`, `scopus_academic_topic_labeling.csv` |

For an external-script run, first execute preprocessing and the split cell in section 3.5. Then run the scripts below and return to the notebook to load results. Its coarse-search cell can also write `output/scopus_research_landscape/lda/scopus_academic_coarse_k_search.csv` when fitting is enabled. The matched patent/policy files are required for the whole notebook, including its growth comparison.

### `notebooks/5.scopus_research_landscape.ipynb`

**Input**

- `data/scopus/scopus_full.xlsx`
- `data/lda/lens_from_scopus_scholar.xlsx`
- `data/lda/overton_from_scopus_scholar.xlsx`

**Process:** Alternative academic notebook using the same main input/output paths as the ISS variant. Prepare the corpus and display academic growth, model diagnostics, and topic interpretations.

**Output**

| Directory | Filename(s) |
| --- | --- |
| `output/scopus_research_landscape/tables/` | `scopus_publication_growth.csv` |
| `output/scopus_research_landscape/figures/` | `scopus_publication_growth.pdf`, `academic_growth_patent_policy_annotated.pdf`, `scopus_df_vocabulary_retention.pdf`, `scopus_academic_k_search.pdf`, `scopus_academic_k_marginal_improvement.pdf`, `scopus_academic_candidate_coherence.pdf` |
| `output/scopus_research_landscape/lda/r_preprocessing/` | `scopus_academic_processed.csv` |
| `output/scopus_research_landscape/lda/` | `scopus_academic_corpus.csv`, `scopus_academic_raw_dtm_summary.csv`, `scopus_academic_raw_term_frequency.csv`, `scopus_academic_df_diagnostics.csv`, `scopus_academic_final_dtm_summary.csv`, `scopus_academic_final_dtm_terms.csv`, `scopus_academic_final_dtm.csv`, `scopus_academic_final_vocabulary.csv`, `scopus_academic_train_indices.csv`, `scopus_academic_test_indices.csv`, `scopus_academic_topic_labeling.csv` |

Use one academic variant consistently: both write into the same directories. The supplied non-ISS version still contains earlier ML–OPF labels. Later model-result loading cells depend on the standalone outputs below, including extended results unless bypassed for K=2–25.

### `scripts/5.scopus_coarse_k_parallel_iss.py`

**Input**

- `output/scopus_research_landscape/lda/r_preprocessing/scopus_academic_processed.csv`
- `output/scopus_research_landscape/lda/scopus_academic_train_indices.csv`
- `output/scopus_research_landscape/lda/scopus_academic_test_indices.csv`

**Process:** Fit independent short-chain LDA models for the configured `K_VALUES`, using the saved train/test split and per-K resume files.

**Output**

| Directory | Filename(s) |
| --- | --- |
| `output/scopus_research_landscape/lda/coarse_parallel/` | `k_{K:03d}.csv`, `k_{K:03d}.log` |
| `output/scopus_research_landscape/lda/` | `scopus_academic_coarse_k_search.csv`, `scopus_academic_coarse_parallel.log` |

### `scripts/5.scopus_extended_k_parallel_iss.py`

**Input**

- `output/scopus_research_landscape/lda/r_preprocessing/scopus_academic_processed.csv`
- `output/scopus_research_landscape/lda/scopus_academic_train_indices.csv`
- `output/scopus_research_landscape/lda/scopus_academic_test_indices.csv`

**Process:** Fit independent short-chain LDA models for the configured `K_VALUES`, using the saved train/test split and per-K resume files.

**Output**

| Directory | Filename(s) |
| --- | --- |
| `output/scopus_research_landscape/lda/extended_parallel/` | `k_{K:03d}.csv`, `k_{K:03d}.log` |
| `output/scopus_research_landscape/lda/` | `scopus_academic_extended_k_search.csv`, `scopus_academic_extended_parallel.log` |

The extended script is optional and should be skipped when the coarse script already covers every K from 2 through 25.

### `scripts/5.scopus_candidate_evaluation_parallel_iss.py`

**Input**

- `output/scopus_research_landscape/lda/r_preprocessing/scopus_academic_processed.csv`
- `output/scopus_research_landscape/lda/scopus_academic_train_indices.csv`
- `output/scopus_research_landscape/lda/scopus_academic_test_indices.csv`

**Process:** Fit longer-chain candidate models and calculate held-out perplexity, coherence, inter-topic similarity, and topic-level diagnostics.

**Output**

| Directory | Filename(s) |
| --- | --- |
| `output/scopus_research_landscape/lda/candidate_parallel/` | `k_{K:03d}_summary.csv`, `k_{K:03d}_topics.csv`, `k_{K:03d}_top_terms.csv`, `k_{K:03d}.log` |
| `output/scopus_research_landscape/lda/` | `scopus_academic_candidate_evaluation.csv`, `scopus_academic_candidate_topic_diagnostics.csv`, `scopus_academic_candidate_top_terms.csv` |

### `scripts/5.scopus_final_lda_iss.py`

**Input**

- `output/scopus_research_landscape/lda/r_preprocessing/scopus_academic_processed.csv`
- `output/scopus_research_landscape/lda/scopus_academic_corpus.csv`

**Process:** Fit the selected `FINAL_K` on the full academic corpus and attach scholarly metadata to document/topic summaries. This final fit does not use the train/test index files.

**Output**

| Directory | Filename(s) |
| --- | --- |
| `output/scopus_research_landscape/lda/` | `scopus_academic_final_model_summary.csv`, `scopus_academic_final_top_terms.csv`, `scopus_academic_final_document_topics.csv`, `scopus_academic_final_topic_prevalence.csv`, `scopus_academic_final_representative_documents.csv`, `scopus_academic_final_lda.log` |
| `output/scopus_research_landscape/lda/final_model/` | `scopus_academic_final_beta.csv` |

### `notebooks/6.patent_policy_citation_landscape.ipynb`

**Input**

- `data/lda/lens_from_scopus_scholar.xlsx`
- `data/lda/overton_from_scopus_scholar.xlsx`
- `data/scopus/scopus_full.xlsx`
- `data/lens/lenspatcite-export-M_Muna-84f6bb81-32bc-4ec9-97c6-f970293a9752-2026-09-25_01-46-54_patents.xlsx`
- `data/overton/articles-2026-09-30.xlsx`

**Process:** Summarize externally cited publications, journals, authors, countries, overlaps, and citation timelines. The last two inputs are required by the later timeline cells; they are different exports from the DOI-matching inputs.

**Output**

| Directory | Filename(s) |
| --- | --- |
| `output/patent_policy_citation_landscape/tables/` | `top_patent_cited_publications.csv`, `top_policy_cited_publications.csv`, `journal_comparison.csv`, `author_comparison.csv`, `top_patent_authors.csv`, `top_policy_authors.csv`, `country_comparison.csv`, `top_patent_countries.csv`, `top_policy_countries.csv` |
| `output/patent_policy_citation_landscape/figures/` | `patent_policy_pathway_overlap.pdf`, `patent_policy_citation_overlap.pdf`, `patent_policy_temporal_evolution.pdf`, `patent_citation_jurisdiction_timeline.pdf`, `policy_citation_country_timeline.pdf`, `policy_citation_organisation_type_timeline.pdf`, `policy_citation_organisation_sector_timeline.pdf` |

Start this notebook with the kernel working directory in `notebooks/`, or replace its `PROJECT_DIR` with the explicit project root.

### `notebooks/7.academic_patent_policy_theme_comparison.ipynb`

**Input**

- `output/scopus_research_landscape/lda/scopus_academic_final_topic_prevalence.csv`
- `output/scopus_research_landscape/lda/scopus_academic_final_top_terms.csv`
- `output/scopus_research_landscape/lda/scopus_academic_topic_labeling.csv`
- `output/scopus_research_landscape/lda/scopus_academic_final_representative_documents.csv`
- `output/patent_policy_topic_modeling/lda/{corpus}_final_topic_interpretation.csv`

**Process:** Read reviewed topic evidence from all three corpora, apply common macro-theme definitions and topic mappings, aggregate probability-weighted prevalence, and compare thematic similarity and external amplification.

**Output**

| Directory | Filename(s) |
| --- | --- |
| `output/academic_patent_policy_comparison/tables/` | `academic_topics_standardized.csv`, `policy_topics_standardized.csv`, `common_macro_theme_taxonomy.csv`, `academic_topic_common_theme_coding_worksheet.csv`, `academic_topic_to_common_macro_theme.csv`, `academic_common_macro_theme_summary.csv`, `academic_patent_policy_common_macro_themes.csv`, `academic_patent_policy_common_macro_themes_long.csv`, `academic_patent_policy_pairwise_similarity.csv`, `academic_patent_policy_theme_consistency.csv`, `academic_patent_policy_external_amplification.csv` |
| `output/academic_patent_policy_comparison/figures/` | `academic_patent_policy_thematic_landscape.pdf`, `academic_patent_policy_bubble_matrix.pdf` |
| `output/patent_policy_topic_modeling/lda/` | `patent_policy_macro_theme_table.csv` |

The setup also checks for `output/patent_policy_topic_modeling/lda/patent_policy_macro_theme_table.csv`, but the supplied notebook rebuilds this table rather than reading it as topic evidence. Saving it overwrites the same filename used by notebook 3. This notebook explicitly sets `PROJECT_DIR = Path.home() / "project" / "review_paper_fahruddin"`; change that setup value if your checkout is elsewhere.

## Rerunning LDA for K 2 to 25

`range(2, 26)` is inclusive of 2 through 25: **24 models per corpus**. Select a final K from the new diagnostics and topic inspection; do not assume the earlier academic K=50 or patent/policy K=20 still applies. Short-chain screening and long-chain candidate evaluation answer different questions, so the second run is substantially more expensive.

### A. Academic Scopus: prepare inputs

In the supplied `5.scopus_research_landscape_iss(1).ipynb`, run sections **3.1–3.4.2 and the train/test split cell in section 3.5** to create `scopus_academic_processed.csv`, `scopus_academic_corpus.csv`, and the train/test index CSVs. Confirm all are aligned and `FINAL_MIN_DOC_FREQ` matches your processed corpus. The standalone scripts currently hard-code `MIN_DOC_FREQ = 254`; keep it in agreement with the notebook's calculated value, especially if the number of documents or preprocessing has changed.

The notebook's section **3.5** sets `K_COARSE` (cell containing `TRAIN_FRACTION = 0.80`) and its `RUN_COARSE_SEARCH = False` cell loads existing results. If you are using the standalone coarse script, keep that flag `False`, set the notebook's `K_COARSE = list(range(2, 26))` for consistency, and run the separate script as shown below. The `K` values are independently hard-coded in each standalone script.

### B. Academic Scopus: change the standalone scripts

In `scripts/5.scopus_coarse_k_parallel_iss.py`, replace its `K_VALUES` block with:

```python
K_VALUES = list(range(2, 26))
MAX_WORKERS = 3  # adjust to resources allocated to this job
```

In `scripts/5.scopus_candidate_evaluation_parallel_iss.py`, replace its `K_VALUES` block with:

```python
K_VALUES = list(range(2, 26))
MAX_WORKERS = 3  # long-chain models need more resources
```

The extended script defaults to K=70,80,90: **do not run it** for this K=2–25 search. In `5.scopus_research_landscape_iss(1).ipynb`, the **“Load extended K-search results computed on ISS”** cell currently raises `FileNotFoundError` if that CSV is absent. For this run, replace that cell with:

```python
# All requested K values were evaluated by the coarse script.
extended_results = pd.DataFrame(columns=["K", "Perplexity"])
```

Then the notebook can combine the newly saved coarse results with this empty table. In its **3.6 candidate-evaluation configuration** cell, set:

```python
K_CANDIDATES = list(range(2, 26))
```

Review cells that explicitly display old candidates (for example, `show_candidate_topics(60)`) and replace their K with a value in the new range. Re-run the notebook's diagnostic/plot cells only after the standalone CSVs exist. Check the displayed K lists: each relevant combined result should contain exactly 2,3,...,25, with no older K values mixed in.

### C. Patent-cited and policy-cited models: edit both Python *and* R K lists

In the supplied `3.topic_modeling_patent_policy(3).ipynb`, the K lists appear twice: in Python configuration cells and as **literal `K_VALUES <- c(...)` inside generated R backend code**. The Python lists are mostly used for display/checking; editing only them does not change what the R backend fits. A manageable short-chain split is:

| Notebook section | Python configuration | Generated R backend |
| --- | --- | --- |
| **5.2 coarse** | `K_COARSE = [2, 5, 10, 15, 20, 25]` | In the cell headed `5.2.1 Coarse-Search R Backend`, set `K_VALUES <- c(2, 5, 10, 15, 20, 25)`. |
| **5.3 extended** | `K_EXTENDED = [k for k in range(2, 26) if k not in K_COARSE]` | In `5.3.1 Extended-Search R Backend`, set `K_VALUES <- setdiff(2:25, c(2, 5, 10, 15, 20, 25))`. |
| **5.4 candidate evaluation** | `K_CANDIDATES = list(range(2, 26))` | In `5.4.1 Candidate-Evaluation Backend`, set `K_VALUES <- 2:25`. |

Run the backend-creation cells again after editing them, **before** running their matching model cells. The coarse plus extended results cover 2–25 once; the long-chain candidate evaluation covers 2–25 independently. Update the old notebook text that says only K=50 and K=60 are extended. Its `show_candidate_topics(..., 20/30)` calls should use inspected K values within 2–25.

The patent/policy `FINAL_K = {"patent_cited": 20, "policy_cited": 20}` cell is a **separate final-model choice**. Change it only after reviewing the new candidate results. The two corpora can have different final K values.

### D. Choose final K, refit, and reinterpret

Use lower held-out perplexity and topic similarity, higher coherence, and the actual top terms/representative publications to choose a useful resolution. There is no rule that final K must be 25. Then:

1. Change `FINAL_K = <selected academic K>` in `scripts/5.scopus_final_lda_iss.py` (and the academic notebook's final-configuration cell if continuing to use that notebook). Run the standalone final script **once** after selection.
2. Change the patent/policy notebook's `FINAL_K` dictionary to the separately selected values and run its final LDA and topic interpretation sections.
3. Rebuild **all manual topic-label and macro-theme mappings** against the newly fitted topics. Topic numbers are model-specific; a topic numbered 7 in a K=20 model is not guaranteed to have the same meaning in a new K=12 model. In the academic ISS notebook, edit the `topic_labels` and `topic_macro_themes` cells and replace `assert set(topic_macro_themes) == set(range(1, 51))` with `assert set(topic_macro_themes) == set(range(1, FINAL_K + 1))`. In the patent/policy notebook, edit `patent_topic_labels`, `policy_topic_labels`, `patent_macro_theme_map`, and `policy_macro_theme_map` after reviewing new terms and representative titles.
4. Revisit the common-theme mapping dictionaries in notebook 7 (`PATENT_TOPIC_TO_COMMON_MACRO`, `POLICY_TOPIC_TO_COMMON_MACRO`, `ACADEMIC_TOPIC_GROUPS`). Its current lists cover 50/20/20 topics and will be invalid for different final K. Change hard-coded document counts and remaining ML–OPF/K=50/K=20 wording in reporting notebooks and manuscript tables.

**Existing result files:** The standalone coarse/candidate scripts resume from per-K CSVs when they look complete; they do not check whether the underlying corpus, vocabulary, random split, or settings changed. When those inputs change, archive the previous `coarse_parallel/` and `candidate_parallel/` directories (or use a new output directory) before starting, so old results are not silently reused. Back up old final outputs if you need them: the final run writes to the same filenames. The patent/policy notebook also writes its earlier named results to the same output hierarchy, so archive the prior output for a clean comparison.

## Running the Scopus scripts on Windows

After setting `RSCRIPT`, preparing the corpus and split files, and editing K to 2–25, use a Conda-enabled PowerShell terminal:

```powershell
conda activate forecasting-lda
Set-Location 'M:\projects_latex\review_paper_fahruddin'
New-Item -ItemType Directory -Force -Path logs
python -u scripts/5.scopus_coarse_k_parallel_iss.py 2>&1 | Tee-Object -FilePath logs/scopus_coarse_k2_25.log
```

After it finishes, check `$LASTEXITCODE` and inspect the resulting CSV. Then run candidate evaluation:

```powershell
python -u scripts/5.scopus_candidate_evaluation_parallel_iss.py 2>&1 | Tee-Object -FilePath logs/scopus_candidate_k2_25.log
```

After choosing and setting `FINAL_K`, run:

```powershell
python -u scripts/5.scopus_final_lda_iss.py 2>&1 | Tee-Object -FilePath logs/scopus_final_lda.log
```

Keep the terminal open and prevent sleep while models run. Monitor the log from another PowerShell terminal with:

```powershell
Get-Content 'M:\projects_latex\review_paper_fahruddin\logs\scopus_coarse_k2_25.log' -Tail 30 -Wait
```

Use Task Manager to inspect Python/R CPU and memory use. Set `MAX_WORKERS` to suit available RAM and CPUs. Run the scripts as files rather than pasting the multiprocessing code into notebook cells. The input/output filenames and K-selection procedure are the same on Windows and ISS.

## Running the Scopus scripts on ISS / Linux

After activating the environment, from the **repository root**:

```bash
mkdir -p logs
python -u scripts/5.scopus_coarse_k_parallel_iss.py 2>&1 | tee logs/scopus_coarse_k2_25.log
python -u scripts/5.scopus_candidate_evaluation_parallel_iss.py 2>&1 | tee logs/scopus_candidate_k2_25.log
# After choosing FINAL_K in the final script:
python -u scripts/5.scopus_final_lda_iss.py 2>&1 | tee logs/scopus_final_lda.log
```

Do **not** start all three together: candidate evaluation follows preparation and coarse inspection; the final full-corpus model follows selection. The standalone scripts generate their R worker files automatically, so you run the `.py` files, not the generated `.R` files. The patent/policy LDA procedure in the supplied repository is notebook-driven: open `3.topic_modeling_patent_policy.ipynb` in JupyterLab and execute its stages in order. Notebook 4 reads its saved output; it does not launch a new fit.

### Optional tmux example for K=2–25 (ISS / Linux only)

Use tmux if the compute environment allows a long interactive job. From the project root:

```bash
tmux new -s scopus_k2_25
```

Inside that session:

```bash
conda activate forecasting-lda  # or: conda activate pytorch
cd ~/project/review_paper_fahruddin
mkdir -p logs
set -o pipefail
python -u scripts/5.scopus_coarse_k_parallel_iss.py 2>&1 | tee logs/scopus_coarse_k2_25.log
```

Detach without stopping the job with **Ctrl+B, then D**. From another shell:

```bash
tmux ls
tmux attach -t scopus_k2_25
tail -f logs/scopus_coarse_k2_25.log
pgrep -afu "$USER" '5.scopus_coarse_k_parallel_iss.py|fit_single_k_scopus.R'
```

After the coarse run finishes, start the candidate script in the same session (or a new session such as `scopus_candidates_k2_25`):

```bash
python -u scripts/5.scopus_candidate_evaluation_parallel_iss.py 2>&1 | tee logs/scopus_candidate_k2_25.log
```

Check `output/scopus_research_landscape/lda/scopus_academic_coarse_k_search.csv` and `.../scopus_academic_candidate_evaluation.csv` before selecting final K. Per-K logs under `coarse_parallel/` and `candidate_parallel/` help diagnose individual failures. `tmux` keeps a session alive after detaching; follow cluster policy about whether the work belongs on a compute node rather than a login node.

## Reproducibility checks

- Confirm that Scopus, Lens, and Overton input paths and column names match the current exports.
- Confirm the academic processed text and metadata have the same document count and ordering; reuse the corresponding train/test index files only for that corpus.
- For the K=2–25 run, verify that the combined coarse and candidate CSVs each contain the expected `K` values exactly once per corpus. Inspect errors in the per-K logs before interpreting a partial result.
- Confirm each selected final model reports the chosen K and its topic prevalence sums to approximately 100% before rounding.
- Reinterpret topic IDs after **every** new final fit, then update manuscript captions, table values, figures, and cross-corpus harmonization. Previously exported forecasting or ML–OPF topic interpretations are not a substitute for the new model's labels.

