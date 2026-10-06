# Forecasting-related research landscape

This repository analyzes forecasting-related scholarly publications from Scopus and the Scopus publications cited by patents (Lens) or policy documents (Overton). The workflows match publications by DOI, build three text corpora, fit separate latent Dirichlet allocation (LDA) models, interpret their topics, and compare academic and external-citation themes. Patent and policy exports supply citation metadata; the matched Scopus records supply the scholarly titles and abstracts used for LDA.

> **Rerun notice:** The checked-in notebooks and scripts were developed with earlier topic-number searches and some still contain ML–OPF wording, fixed `K=50`/`K=20` topic labels, and old macro-theme mappings. For a new search over **every integer K from 2 through 25**, follow [Rerunning LDA for K 2 to 25](#rerunning-lda-for-k-2-to-25) before interpreting or publishing any results. Changing the candidate K values does **not** automatically change the final model.

## Requirements and installation

- Git, Python with Conda (or a compatible environment manager), R, and JupyterLab.
- Python packages used by these files: `pandas`, `numpy`, `scipy`, `matplotlib`, `matplotlib-venn`, and `openpyxl` (for `.xlsx` files).
- R packages used by the generated R backends: `tm`, `SnowballC`, `slam`, and `topicmodels`.
- `tmux` is optional for keeping long commands attached to a server session. On a shared cluster, use the computing resources and job scheduler according to the site's rules.

On a personal Ubuntu machine with administrator access, Git and tmux can be installed with:

```bash
sudo apt update
sudo apt install git tmux
```

On a managed cluster, use modules or ask the administrator for Git/tmux instead of assuming `sudo` access. If Conda is available, create a dedicated Python/R environment:

```bash
conda create -n forecasting-lda -c conda-forge \
  python=3.11 pandas numpy scipy matplotlib matplotlib-venn openpyxl \
  jupyterlab r-base r-tm r-snowballc r-slam r-topicmodels
conda activate forecasting-lda
which python
which Rscript
Rscript -e 'sapply(c("tm","SnowballC","slam","topicmodels"), requireNamespace, quietly=TRUE)'
```

The last command should report `TRUE` for each R package. The standalone Scopus scripts and the modeling notebooks currently specify a **hard-coded** `RSCRIPT = Path(r"/nfs/mfirdausi/miniconda3/envs/pytorch/bin/Rscript")`. If this is not the value of `which Rscript` in your active environment, edit `RSCRIPT` in the four `5.scopus_*_iss.py` scripts and in the setup cells of the academic and patent/policy modeling notebooks. Do this before fitting models.

From the repository root, for example:

```bash
git clone https://github.com/mdzalfirdausi/review_paper_fahruddin.git
cd review_paper_fahruddin
git status
jupyter lab
```

If the repository is private, configure GitHub access before cloning. See the [Git installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), [Jupyter installation guide](https://jupyter.org/install), and [tmux getting started guide](https://github.com/tmux/tmux/wiki/Getting-Started).

## Expected layout and data

Run the Python scripts from the repository root. They determine `PROJECT_DIR` as the parent of `scripts/`. Open notebooks with the repository root as the working directory or from `notebooks/`; most modeling notebooks detect either location. **The citation-landscape notebook is an exception:** its `PROJECT_DIR = Path.cwd().resolve().parent` assumes the notebook kernel starts in `notebooks/`. Set `PROJECT_DIR` explicitly if starting it at the root.

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

From the repository root, after placing the exports and checking their configured filenames:

```bash
python scripts/extract_lens_dois.py
python scripts/extract_patent_cited_scholar.py
python scripts/extract_overton_cited_scholar.py
```

Optionally run `python scripts/0.doi_filter_overton.py` first when working from its configured raw CSV. The patent/policy topic-modeling notebook and citation-landscape notebook consume the two matched `.xlsx` files; the academic notebook consumes `data/scopus/scopus_full.xlsx`.

## Rerunning LDA for K 2 to 25

`range(2, 26)` is inclusive of 2 through 25: **24 models per corpus**. Select a final K from the new diagnostics and topic inspection; do not assume the earlier academic K=50 or patent/policy K=20 still applies. Short-chain screening and long-chain candidate evaluation answer different questions, so the second run is substantially more expensive.

### A. Academic Scopus: prepare inputs

In the supplied `5.scopus_research_landscape_iss(1).ipynb`, run sections **3.1–3.4.2** to create `scopus_academic_processed.csv`, `scopus_academic_corpus.csv`, and the train/test index CSVs. Confirm all are aligned and `FINAL_MIN_DOC_FREQ` matches your processed corpus. The standalone scripts currently hard-code `MIN_DOC_FREQ = 254`; keep it in agreement with the notebook's calculated value, especially if the number of documents or preprocessing has changed.

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

## Running the Scopus scripts on ISS

After activating the environment, from the **repository root**:

```bash
mkdir -p logs
python -u scripts/5.scopus_coarse_k_parallel_iss.py 2>&1 | tee logs/scopus_coarse_k2_25.log
python -u scripts/5.scopus_candidate_evaluation_parallel_iss.py 2>&1 | tee logs/scopus_candidate_k2_25.log
# After choosing FINAL_K in the final script:
python -u scripts/5.scopus_final_lda_iss.py 2>&1 | tee logs/scopus_final_lda.log
```

Do **not** start all three together: candidate evaluation follows preparation and coarse inspection; the final full-corpus model follows selection. The standalone scripts generate their R worker files automatically, so you run the `.py` files, not the generated `.R` files. The patent/policy LDA procedure in the supplied repository is notebook-driven: open `3.topic_modeling_patent_policy.ipynb` in JupyterLab and execute its stages in order. Notebook 4 reads its saved output; it does not launch a new fit.

### Optional tmux example for K=2–25

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

