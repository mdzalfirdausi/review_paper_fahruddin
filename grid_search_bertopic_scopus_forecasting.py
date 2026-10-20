#!/usr/bin/env python3
"""Portable CLI conversion of the supplied orchestration notebook.

This is a launcher, not a replacement implementation of the two original
research engines. Obtain run_grid_v2.py, evaluate_grid_v2.py and their local
dependencies before training. Run --action inventory first. See RUN_GUIDE.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

TOOLS_RELATIVE = Path("Reports/2026-10-17_bertopic_tuning/grid_v2_outlier8")
REVISED = {
    "NN": [10, 20, 30, 40, 50, 60],
    "NC": [10, 20, 30],
    "MCS": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    "MS": [10, 20, 30, 40, 50, 60],
    "SEL": ["eom", "leaf"],
}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, type=Path, help="Existing local project folder, not a SharePoint URL")
    p.add_argument("--tools-dir", type=Path, help="Engine directory; defaults to ROOT/Reports/2026-10-17_bertopic_tuning/grid_v2_outlier8")
    p.add_argument("--db", default="scopus", help="Database key accepted by the original engines")
    p.add_argument("--action", required=True, choices=["inventory", "check", "grid", "run", "progress", "collect", "rank", "stability", "final", "top3"])
    p.add_argument("--profile", choices=["revised3888", "notebook"], default="revised3888", help="Validate engine constants against revised or original notebook grid")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--batch-id", type=int, help="Zero-based batch ID shown by --action grid; revised profile: 0..17")
    selection.add_argument("--embedder", help="Actual engine alias printed by --action grid")
    p.add_argument("--nn", type=int, nargs="+", help="Neighbor values when selecting by --embedder")
    p.add_argument("--all", action="store_true", help="Explicitly allow running every batch on this machine")
    p.add_argument("--from-results", action="append", type=Path, default=[], help="collect: source grid_v2 folder copied from another machine; repeatable")
    p.add_argument("--input-files", nargs="+", type=Path, help="run: complete list of data/config inputs used by the backend; hashed for cross-machine consistency")
    p.add_argument("--part", type=int, default=0, help="stability: zero-based partition")
    p.add_argument("--parts", type=int, default=1, help="stability: total number of partitions")
    p.add_argument("--threads", type=int, default=2, help="NUMBA threads; numerical libraries use one thread")
    p.add_argument("--extra-pythonpath", action="append", type=Path, default=[], help="Optional extra import folder; repeatable")
    p.add_argument("--dry-run", action="store_true", help="Print backend commands without executing them")
    return p


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def inventory(args, here):
    print("Project root:", args.root)
    print("Engine folder:", here)
    print("Python:", sys.executable)
    for name in ["run_grid_v2.py", "evaluate_grid_v2.py"]:
        path = here / name
        print(("FOUND: " if path.is_file() else "MISSING: ") + str(path))
    print("Download that Reports folder AND its parent Python helpers.")
    print("Download the data, cached embeddings, vocabulary/config files referenced by those helpers.")
    print("The notebook alone does not reveal their exact filenames. No training was started.")


def load_engine(name, args, here):
    if not (here / (name + ".py")).is_file():
        raise ValueError(f"Missing {here / (name + '.py')}. Download the original engine and its dependencies.")
    try:
        mod = importlib.import_module(name)
    except ImportError as exc:
        raise ValueError(f"Cannot import {name}: {exc}. Copy its local helpers/install its original dependencies.") from exc
    backend_root = getattr(mod, "ROOT", None)
    if backend_root is not None and Path(backend_root).resolve() != args.root:
        raise ValueError(f"{name}.ROOT is {backend_root!s}, but --root is {args.root}. Update that engine's ROOT and any helper roots; the launcher does not rewrite unseen source code.")
    return mod


def make_grid(g, profile):
    needed = ["EMBS", "NN", "NC", "MCS", "MS", "SEL", "grid_id"]
    missing = [name for name in needed if not hasattr(g, name)]
    if missing:
        raise ValueError("Runner lacks notebook API: " + ", ".join(missing))
    target = REVISED if profile == "revised3888" else {
        "NN": list(range(10, 51, 5)), "NC": list(range(10, 51, 5)),
        "MCS": [0.35, 0.70, 1.0, 1.35, 1.70, 2.0, 2.35, 2.70, 3.0],
        "MS": list(range(10, 61, 10)), "SEL": ["eom", "leaf"],
    }
    for key, values in target.items():
        if list(getattr(g, key)) != values:
            raise ValueError(f"Grid mismatch: run_grid_v2.{key}={list(getattr(g, key))}; expected {values}. Update the original engine constant before running. Also check for duplicated grid constants in the evaluator/helpers.")
    if len(g.EMBS) != 3:
        raise ValueError("Expected three embedding models from the original engine.")
    rows, batches = [], []
    for batch_id, (emb, nn) in enumerate(itertools.product(g.EMBS, g.NN)):
        batches.append({"batch_id": batch_id, "embedder": emb, "n_neighbors": nn})
        for nc, mcs, ms, sel in itertools.product(g.NC, g.MCS, g.MS, g.SEL):
            rows.append({"Grid_id": str(g.grid_id(emb, nn, nc, mcs, ms, sel)), "batch_id": batch_id,
                         "embedder": emb, "n_neighbors": nn, "n_components": nc,
                         "mcs_share_pct": mcs, "min_samples": ms, "selection": sel})
    if len({r["Grid_id"] for r in rows}) != len(rows):
        raise ValueError("Engine generated duplicate Grid_id values.")
    expected = 3888 if profile == "revised3888" else 26244
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} settings, found {len(rows)}")
    return rows, batches


def scan(results, expected):
    found, errors = {}, []
    for path in sorted(results.glob("*/summary.json")):
        gid = path.parent.name
        if gid not in expected:
            errors.append(f"Unexpected configuration: {gid}")
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(record, dict):
                raise ValueError("summary is not a JSON object")
            if "Grid_id" in record and str(record["Grid_id"]) != gid:
                raise ValueError("Grid_id disagrees with directory name")
            if record.get("error") or record.get("Error") or str(record.get("status", "")).lower() in {"failed", "error"}:
                raise ValueError("summary reports failure")
            found[gid] = record
        except (ValueError, OSError) as exc:
            errors.append(f"{path}: {exc}")
    return found, errors


def require_complete(results, expected):
    found, errors = scan(results, expected)
    missing = sorted(expected - found.keys())
    if missing or errors:
        raise ValueError(f"Cannot rank: {len(found)}/{len(expected)} readable summaries, {len(missing)} missing, {len(errors)} invalid/unexpected. Run progress and resolve these first.")
    return found


def experiment_identity(args, here, rows):
    if not args.input_files:
        raise ValueError("run requires --input-files listing all backend data/config inputs, so results from different machines can be checked before merging.")
    inputs = []
    for path in args.input_files:
        path = path if path.is_absolute() else args.root / path
        if not path.is_file():
            raise ValueError(f"Input file not found: {path}")
        inputs.append({"name": path.name, "sha256": digest(path)})
    return {"db": args.db, "profile": args.profile,
            "grid_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
            "inputs": sorted(inputs, key=lambda x: (x["name"], x["sha256"])),
            "engine_sha256": {n: digest(here / n) for n in ["run_grid_v2.py", "evaluate_grid_v2.py"]}}


def save_identity(results, identity):
    path = results / "experiment_identity.json"
    if not path.exists() and any(results.glob("*/summary.json")):
        raise ValueError("Destination contains unmarked results from an older run. Use a clean project/results directory; do not silently mix experiments.")
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Existing results belong to a different experiment. Use a separate project/output directory.")
    results.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")


def collect(args, results, expected, rows, here):
    if not args.from_results:
        raise ValueError("collect requires one or more --from-results folders")
    # Preflight every source and all file conflicts before copying any results.
    sources = [p.expanduser().resolve() for p in args.from_results]
    identities = []
    for root in ([results] if (results / "experiment_identity.json").exists() else []) + sources:
        marker = root / "experiment_identity.json"
        if not marker.is_file():
            raise ValueError(f"Missing {marker}; collect only outputs produced with this launcher's run action.")
        identity = json.loads(marker.read_text())
        expected_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        if identity.get("db") != args.db or identity.get("profile") != args.profile or identity.get("grid_sha256") != expected_hash:
            raise ValueError(f"{marker} belongs to a different database/grid/profile.")
        local_engines = {n: digest(here / n) for n in ["run_grid_v2.py", "evaluate_grid_v2.py"]}
        if identity.get("engine_sha256") != local_engines:
            raise ValueError(f"{marker}: engine files do not match the ranking machine's engines.")
        identities.append(identity)
    if any(identity != identities[0] for identity in identities):
        raise ValueError("Experiments differ: input/grid/engine hashes do not match. Do not merge these results.")
    planned, fingerprints = [], {}
    for source in sources:
        if source == results:
            raise ValueError("A source results folder cannot equal the destination.")
        found, errors = scan(source, expected)
        if errors:
            raise ValueError("\n".join(errors[:20]))
        if not found:
            raise ValueError(f"No configuration summaries found in {source}")
        for gid in found:
            for src in (source / gid).rglob("*"):
                if not src.is_file():
                    continue
                rel = src.relative_to(source)
                checksum = digest(src)
                dst = results / rel
                if (rel in fingerprints and fingerprints[rel] != checksum) or (dst.exists() and digest(dst) != checksum):
                    raise ValueError(f"Conflicting result file: {rel}; no conflicting result is overwritten.")
                fingerprints[rel] = checksum
                planned.append((src, dst))
    save_identity(results, identities[0])
    for src, dst in planned:
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    print("Collected result folders into", results)


def run_command(command, args):
    print("Command:", subprocess.list2cmdline([str(x) for x in command]), flush=True)
    if not args.dry_run:
        subprocess.run([str(x) for x in command], cwd=args.root, check=True)


def main():
    args = parser().parse_args()
    args.root = args.root.expanduser().resolve()
    if not args.root.is_dir():
        raise ValueError(f"Project folder does not exist: {args.root}. Download/sync the shared files locally and pass their actual folder with --root.")
    if args.threads < 1 or args.parts < 1 or not 0 <= args.part < args.parts:
        raise ValueError("Require threads >= 1 and 0 <= part < parts.")
    if Path(args.db).name != args.db or "/" in args.db or "\\" in args.db or args.db in {".", ".."}:
        raise ValueError("--db must be a database key, not a path")
    here = (args.tools_dir or args.root / TOOLS_RELATIVE).expanduser().resolve()
    if args.action == "inventory":
        inventory(args, here)
        return
    os.chdir(args.root)
    os.environ["BERTOPIC_PROJECT_ROOT"] = str(args.root)
    imports = [here, here.parent, args.root, args.root / "Codes", *[p.expanduser().resolve() for p in args.extra_pythonpath]]
    sys.path[:0] = [str(p) for p in imports]
    os.environ["PYTHONPATH"] = os.pathsep.join([*[str(p) for p in imports], os.environ.get("PYTHONPATH", "")])
    os.environ["NUMBA_NUM_THREADS"] = str(args.threads)
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        os.environ[key] = "1"
    g = load_engine("run_grid_v2", args, here)
    rows, batches = make_grid(g, args.profile)
    expected = {row["Grid_id"] for row in rows}
    results = args.root / "Codes/output/bertopic_tuning" / args.db / "grid_v2"
    cli_dir = results.parent / ("cli_" + args.profile)
    if args.action in {"check", "grid"}:
        if args.action == "check":
            e = load_engine("evaluate_grid_v2", args, here)
            for name in ["step_rank", "step_final", "step_top3"]:
                if not callable(getattr(e, name, None)):
                    raise ValueError(f"Evaluator lacks {name}")
            print("Engine imports/API/grid checked. Backend data paths, scoring implementation and full execution still require the original files and environment.")
        else:
            for b in batches:
                b["configurations"] = sum(r["batch_id"] == b["batch_id"] for r in rows)
            write_csv(cli_dir / "grid_manifest.csv", rows)
            write_csv(cli_dir / "batches.csv", batches)
            for b in batches:
                print(b)
            print("Manifests:", cli_dir)
        print("Configurations:", len(rows), "| batches:", len(batches), "| results:", results)
        return
    if args.action == "run":
        if not (here / "evaluate_grid_v2.py").is_file():
            raise ValueError("Download evaluate_grid_v2.py too; its hash is required for experiment identity.")
        identity = experiment_identity(args, here, rows)
        if args.batch_id is not None:
            if not 0 <= args.batch_id < len(batches):
                raise ValueError(f"--batch-id must be 0..{len(batches)-1}")
            selected = [batches[args.batch_id]]
        elif args.embedder:
            if args.embedder not in g.EMBS:
                raise ValueError(f"Unknown embedder. Actual aliases: {list(g.EMBS)}")
            if args.nn and any(nn not in g.NN for nn in args.nn):
                raise ValueError(f"--nn must contain values from {list(g.NN)}")
            selected = [b for b in batches if b["embedder"] == args.embedder and (not args.nn or b["n_neighbors"] in args.nn)]
        elif args.all:
            selected = batches
        else:
            raise ValueError("Select --batch-id, --embedder [--nn ...], or explicitly --all")
        if args.nn and not args.embedder:
            raise ValueError("--nn is used only with --embedder")
        if not args.dry_run:
            save_identity(results, identity)
        for batch in selected:
            run_command([sys.executable, "-X", "utf8", here / "run_grid_v2.py", "--db", args.db,
                         "--embedder", batch["embedder"], "--nn", batch["n_neighbors"]], args)
        return
    if args.action == "collect":
        collect(args, results, expected, rows, here)
    if args.action in {"progress", "collect"}:
        found, errors = scan(results, expected)
        print(f"{len(found):,}/{len(expected):,} readable configuration summaries")
        write_csv(cli_dir / "missing.csv", [r for r in rows if r["Grid_id"] not in found])
        write_csv(cli_dir / "raw_results.csv", [{**r, "summary": found[r["Grid_id"]]} for r in rows if r["Grid_id"] in found])
        for err in errors[:20]:
            print("ERROR:", err)
        print("Raw summary export and missing list:", cli_dir)
        if errors:
            raise ValueError("Invalid/unexpected summaries found; resolve before ranking")
        return
    require_complete(results, expected)
    if args.action == "stability":
        run_command([sys.executable, "-X", "utf8", here / "evaluate_grid_v2.py", "stability", "--db", args.db,
                     "--part", args.part, "--parts", args.parts], args)
        return
    e = load_engine("evaluate_grid_v2", args, here)
    tables = Path(e.TABLES).resolve()
    if not tables.is_relative_to(args.root):
        raise ValueError(f"Evaluator TABLES is outside --root: {tables}; update the engine paths.")
    if args.dry_run:
        print(f"Would call evaluate_grid_v2.step_{args.action}({args.db!r}); tables: {tables}")
        return
    getattr(e, "step_" + args.action)(args.db)
    print("Finished", args.action, "| evaluator tables:", tables)
    if args.action == "final":
        print("Original notebook expects", tables / f"{args.db}_top20.csv")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
