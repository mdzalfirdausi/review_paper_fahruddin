# BERTopic grid search on several machines

This package converts the uploaded notebook's orchestration into a Python CLI.
It DOES NOT contain the original training/scoring implementations. These live in
`run_grid_v2.py` and `evaluate_grid_v2.py`, imported by the uploaded notebook.
Neither engine was provided, so full training, metric definitions, backend data
paths, engine resume behaviour and evaluator compatibility cannot be verified here.
The original notebook is left unchanged.

## What to download from the shared online directory

The screenshot shows the shared project `Journal SLR_Scopus_Overton`. Download or
sync files to a REAL LOCAL folder on each machine. An online folder URL is not a
Windows filesystem directory. The error in the screenshot occurs because
`D:/Journal SLR_Scopus_Overton` does not exist on that machine.

Confirmed by the uploaded notebook:

| Item | Original location relative to the project root | Why |
|---|---|---|
| Grid engine | `Reports/2026-10-17_bertopic_tuning/grid_v2_outlier8/run_grid_v2.py` | Runs training and writes configuration summaries |
| Evaluation engine | `Reports/2026-10-17_bertopic_tuning/grid_v2_outlier8/evaluate_grid_v2.py` | Preliminary ranking, stability, final ranking and top-three tables |
| Parent helper directory | `Reports/2026-10-17_bertopic_tuning/` | The notebook adds this parent directory to its import path |
| Original result directory | `Codes/output/bertopic_tuning/scopus/grid_v2/` | Contains one directory per Grid_id, each with summary.json |

Download the COMPLETE `Reports/2026-10-17_bertopic_tuning/` directory, rather than
only the notebook. Check the two engines and their imported helper scripts for
all input paths. Copy the exact Scopus corpus, prepared-text files, vocabulary,
stopword/config files and embedding caches that THEY load. Do not substitute
another Scopus export just because its filename is similar. The notebook itself
does not identify these input filenames or their columns.

`Clean Database/` and `Codes/` are visible in the screenshot, but their internal
contents are not. They may contain required inputs/helpers; this cannot be
confirmed from the screenshot. Copy them if referenced by the engines.

Use the same document order, document contents, preprocessing, embedding-model
revision and dependency versions on every machine. A GPU is useful for creating
embeddings; do not mix different clustering implementations within one search.

The old Windows interpreter directory is NOT a portable environment. Recreate
the original Python environment on each machine. On the working machine, export:

```powershell
python -m pip freeze > requirements-original.txt
```

On each new machine, use a compatible Python version (this launcher needs 3.10+),
create/activate a virtual environment, then install that requirements file:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-original.txt
```

The notebook explicitly requires pandas and its engines; other dependencies
must be obtained from those engines/the working environment. Its comment pins
snowballstemmer to 2.2.0; preserve that version if reproducing those outputs.
Embedding models must be downloaded or their caches copied before offline runs.

## Make backend paths portable once

The launcher accepts `--root` and uses the ACTIVE Python interpreter, rather
than the old D-drive interpreter. It removes the notebook's need for the
`.claude/.../checkpoint.py` wrapper. The old PowerShell launcher and watchdog
are not required for this launcher. Resume remains the original engine's job.

However, `--root` cannot automatically correct hard-coded paths inside unseen
backend scripts. Update their root definitions and imported helpers once, using
the SAME code on all machines. For example, replace a fixed ROOT assignment with:

```python
import os
from pathlib import Path

ROOT = Path(os.environ["BERTOPIC_PROJECT_ROOT"]).expanduser().resolve()
```

The launcher sets that environment variable before importing/running the engines.
Apply this only where an engine actually defines its project ROOT, keeping its
existing derived paths. Check any other absolute data/model/interpreter paths.
If the backend defines ROOT, this launcher checks that it agrees with --root.
It cannot discover or repair every arbitrary path in imported code.

Keep engine files byte-identical across machines: the collector checks their
hashes as well as the declared input-file hashes. Do not edit the root to a
different literal on each machine. Use the environment-variable approach above.

## Revised grid: 3,888 configurations

In the original `run_grid_v2.py`, update the grid constants (if they have the
same names used by the notebook):

```python
NN = [10, 20, 30, 40, 50, 60]
NC = [10, 20, 30]
MCS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # percentages, not fractions
MS = [10, 20, 30, 40, 50, 60]
SEL = ["eom", "leaf"]
```

Preserve the engine's three actual EMBS aliases/model identifiers. Also inspect
the evaluator/other helpers for duplicated grid constants or a fixed 26,244
completion assertion, updating them consistently. This launcher validates the
runner's constants; it does not silently monkey-patch them or rewrite research
code. The initial seed remains the backend's original 123; verify this there.

`3 * 6 * 3 * 6 * 6 * 2 = 3,888` configurations. There are 18 batches, one per
embedding-model/neighbour-value pair, and 216 configurations per batch.
Batch IDs use the order of EMBS and NN in the ORIGINAL runner; inspect batches.csv.

The optional `--profile notebook` validates the ORIGINAL uploaded 26,244 grid.
It is not the revised experiment and should use a separate output/project tree.
Never combine the two profiles or old unmarked outputs.

## Initial checks

Place the launcher anywhere convenient. The commands below assume it is in
your terminal's current directory. Replace C:/Research/Journal SLR_Scopus_Overton
with the ACTUAL local downloaded project folder. Quote paths containing spaces.
Use `python` from the activated environment, or its full interpreter path.

```powershell
python grid_search_bertopic_scopus_forecasting.py --help
python grid_search_bertopic_scopus_forecasting.py --root "C:/Research/Journal SLR_Scopus_Overton" --action inventory
python grid_search_bertopic_scopus_forecasting.py --root "C:/Research/Journal SLR_Scopus_Overton" --action check
python grid_search_bertopic_scopus_forecasting.py --root "C:/Research/Journal SLR_Scopus_Overton" --action grid
```

`inventory` works without the research dependencies. `check` imports the engines
and checks their notebook API/grid/root. `grid` writes grid_manifest.csv and
batches.csv under `Codes/output/bertopic_tuning/scopus/cli_revised3888/`.
These checks do not prove the engines can load their backend data.

If your engines are in another directory, append `--tools-dir "ABSOLUTE/PATH"`.
Use `--extra-pythonpath "ABSOLUTE/PATH"` for an additional local helper directory.
Linux commands use the same flags with Linux paths and an activated environment.

## Run assigned work on each machine

Use ALL original input files on EVERY machine. Divide configurations, not papers.
In the commands below the path `RELATIVE/PATH/TO/ACTUAL_CORPUS_FILE` is a deliberate
placeholder: replace it with the real backend input path. Add every other data/
config input after it. The notebook does not expose those filenames.

```powershell
python grid_search_bertopic_scopus_forecasting.py --root "C:/Research/Journal SLR_Scopus_Overton" --action run --batch-id 0 --input-files "RELATIVE/PATH/TO/ACTUAL_CORPUS_FILE" --dry-run
python grid_search_bertopic_scopus_forecasting.py --root "C:/Research/Journal SLR_Scopus_Overton" --action run --batch-id 0 --input-files "RELATIVE/PATH/TO/ACTUAL_CORPUS_FILE"
```

Assign batch IDs 0 through 17 without overlap. Faster machines can take more
unstarted batches. Each launcher invocation runs its selected batches sequentially.
For an entire embedding model, use its ACTUAL alias printed by `--action grid`:

```powershell
python grid_search_bertopic_scopus_forecasting.py --root "C:/Research/Journal SLR_Scopus_Overton" --action run --embedder "ACTUAL_ALIAS" --nn 10 20 --input-files "RELATIVE/PATH/TO/ACTUAL_CORPUS_FILE"
```

Omit --nn to run all neighbour values for that embedder. Explicit --all runs
every batch on this machine. --threads controls NUMBA threads per process; other
numerical libraries are limited to one thread. Avoid oversubscribing RAM/CPUs.

The input flag fingerprints files; it DOES NOT override the original engines'
data-loading paths. Those paths must already point at the same listed inputs.
Use identical basenames and bytes for these declared inputs on each machine.

The launcher reuses the original engine CLI, known from the notebook:
`run_grid_v2.py --db scopus --embedder ALIAS --nn VALUE`.
It does not implement its own embedding/UMAP cache. Whether those are reused
efficiently must be checked in run_grid_v2.py after obtaining it.

## Monitor and collect

```powershell
python grid_search_bertopic_scopus_forecasting.py --root "C:/Research/Journal SLR_Scopus_Overton" --action progress
```

This exports raw JSON summaries into raw_results.csv and lists missing settings
in missing.csv, under the CLI directory above. It counts readable summaries;
the original evaluator remains responsible for metric validity.

Copy each machine's COMPLETE `Codes/output/bertopic_tuning/scopus/grid_v2/`
folder to its own staging location on the ranking machine. Include
experiment_identity.json and all per-configuration files, not only CSV scores.
Do not write to another worker's live output directory. Collect finished results
after the worker has exited. Keep original copies until ranking succeeds.

```powershell
python grid_search_bertopic_scopus_forecasting.py --root "C:/Research/Journal SLR_Scopus_Overton" --action collect --from-results "C:/Collected/machine2/grid_v2" --from-results "C:/Collected/machine3/grid_v2"
```

The collector verifies matching declared input/grid/engine hashes, rejects
unexpected configurations and conflicting duplicate files, and copies entire
configuration folders without overwriting conflicting results. It does not
merge model objects or rewrite absolute paths inside original engine outputs.
If the evaluator uses absolute cached-model/data paths, those must be made
portable or the needed inputs/caches restored before evaluation. Local helper
hashes and package-version equivalence are not automatically checked: preserve
the same helper tree and environment on all machines.

## Rank on ONE machine after collection

```powershell
python grid_search_bertopic_scopus_forecasting.py --root "C:/Research/Journal SLR_Scopus_Overton" --action rank
python grid_search_bertopic_scopus_forecasting.py --root "C:/Research/Journal SLR_Scopus_Overton" --action stability --part 0 --parts 1
python grid_search_bertopic_scopus_forecasting.py --root "C:/Research/Journal SLR_Scopus_Overton" --action final
python grid_search_bertopic_scopus_forecasting.py --root "C:/Research/Journal SLR_Scopus_Overton" --action top3
```

These call the original evaluator stages. rank/final/top3 require all expected
configuration summaries before dispatch. The notebook documents 5-30 eligible
topics, seven preliminary criteria, stability for the global top 60 using seeds
0/1/2, then eight criteria in the final shortlist ranking. Do not calculate local
quartiles separately on workers. Outliers above 8% receive zero points; this is
not a hard exclusion under the documented scoring rule.

The notebook expects these filenames in the ORIGINAL evaluator's TABLES path:

- scopus_ranked_pre.csv
- scopus_ranked.csv
- scopus_top20.csv
- scopus_top3_groups.csv
- scopus_top3_members.csv

The exact TABLES location is not defined in the notebook; it comes from
evaluate_grid_v2.py and is printed by the launcher. The evaluator is responsible
for waiting for/checking stability completion before final ranking. The launcher
cannot independently verify the unseen stability file format.

Stability partition flags are retained, but initially use one ranking machine.
Distributing stability requires checking and transferring its own backend cache/
output files, whose locations are not visible in the notebook. The collection
action above is specifically for the initial grid result directories.

## Verification performed for this conversion

The launcher was syntax-checked and its CLI, grid counts/batches, command
dispatch, grid-mismatch rejection, result collection and completeness checks
were tested using lightweight fixture engines. BERTopic training and real
rankings were NOT run: the two original engines, their inputs and their exact
environment were not attached. Fixture tests are not evidence of engine/data
compatibility. Obtain those files before starting the real search.
