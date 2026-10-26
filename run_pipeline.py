#!/usr/bin/env python3
"""Run the whole CV pipeline in sequence and log every step.

    1. convert.py                PDF            -> mds/<name>.md
    2. headings_hierarchy.py     <name>.md      -> mds/<name>_corrected.md
    3. experience_lines.py       _corrected.md  -> mds/<name>_corrected_experience.md (+ .log.jsonl)
    4. extract_date_intervals.py _experience.md -> output/<name>_corrected_experience.json

Each stage is given explicit file paths for the CVs of this run, so earlier outputs
are never fed back into a stage. A stage whose output already exists is skipped unless
--force is given. Stages 3 and 4 keep only sections whose heading is embedding-close to "work experience".

Usage:
    python run_pipeline.py                      # every PDF in ./cvs
    python run_pipeline.py "cvs/Sandeep Kumar.pdf" other_dir
    python run_pipeline.py --force --prefilter
Logs go to the console and to logs/pipeline_<timestamp>.log.
"""
import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
OLLAMA = "http://localhost:11434"
REQUIRED_MODELS = ["qwen2.5:3b", "qwen2.5:1.5b"]  # the embedding model is added from --embed-model

log = logging.getLogger("pipeline")


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stderr), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return path


def run(cmd: list[str], label: str) -> int:
    """Run a stage script, streaming each output line into the log."""
    log.info("[%s] $ %s", label, " ".join(cmd))
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log.info("[%s] %s", label, line)
    code = proc.wait()
    log.info("[%s] exit code %d in %.1fs", label, code, time.time() - start)
    return code


def check_ollama(extra: list[str]) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5) as r:
            import json
            have = {m["name"] for m in json.load(r)["models"]}
    except (urllib.error.URLError, OSError) as e:
        log.error("cannot reach Ollama at %s: %s", OLLAMA, e)
        return False
    missing = [m for m in REQUIRED_MODELS + extra if m not in have and f"{m}:latest" not in have]
    for m in missing:
        log.error("Ollama model not pulled: %s (ollama pull %s)", m, m)
    return not missing


def collect_pdfs(paths: list[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for p in paths:
        if p.is_dir():
            pdfs += sorted(p.glob("*.pdf"))
        elif p.is_file():
            pdfs.append(p)
        else:
            log.error("not found: %s", p)
    return pdfs


def remove(path: Path) -> None:
    if path.exists():
        path.unlink()
        log.info("removed old %s", path.name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, default=[BASE / "cvs"], help="PDF files or folders (default: ./cvs)")
    ap.add_argument("--md-dir", type=Path, default=BASE / "mds")
    ap.add_argument("--out-dir", type=Path, default=BASE / "output")
    ap.add_argument("--log-dir", type=Path, default=BASE / "logs")
    ap.add_argument("--force", action="store_true", help="redo every stage, replacing earlier outputs")
    ap.add_argument("--threshold", type=float, default=0.50, help="experience_lines threshold")
    ap.add_argument("--prefilter", action="store_true", help="experience_lines: only score lines with a digit/present word")
    ap.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model for the section gate")
    ap.add_argument("--embed-threshold", type=float, default=0.75, help="section gate: min cosine similarity")
    ap.add_argument("--no-section-gate", action="store_true", help="disable the embedding section gate")
    ap.add_argument("--gate-headings", action="store_true", help="experience_lines: model-based heading gate")
    args = ap.parse_args()

    log_path = setup_logging(args.log_dir)
    log.info("pipeline start; log file: %s", log_path)
    py = sys.executable
    gate_args = ["--no-section-gate"] if args.no_section_gate else [
        "--embed-model", args.embed_model, "--embed-threshold", str(args.embed_threshold)]
    md_dir, out_dir = args.md_dir.resolve(), args.out_dir.resolve()
    md_dir.mkdir(parents=True, exist_ok=True)

    pdfs = collect_pdfs(args.paths)
    if not pdfs:
        log.error("no PDFs found")
        return 1
    log.info("%d PDF(s): %s", len(pdfs), ", ".join(p.name for p in pdfs))
    if not check_ollama([] if args.no_section_gate else [args.embed_model]):
        return 1

    t0 = time.time()
    failed: dict[str, str] = {}
    stems = [p.stem for p in pdfs]

    def raw(s): return md_dir / f"{s}.md"
    def corrected(s): return md_dir / f"{s}_corrected.md"
    def experience(s): return md_dir / f"{s}_corrected_experience.md"
    def result(s): return out_dir / f"{s}_corrected_experience.json"

    if args.force:
        for s in stems:
            for f in (raw(s), corrected(s), experience(s), md_dir / f"{s}_corrected_experience.log.jsonl", result(s)):
                remove(f)

    # ---- stage 1: PDF -> markdown
    log.info("===== STAGE 1/4: convert PDF -> markdown")
    todo = [p for p in pdfs if not raw(p.stem).exists() and not corrected(p.stem).exists()]
    for p in pdfs:
        if p not in todo:
            log.info("skip convert %s: markdown already exists", p.name)
    if todo:
        with tempfile.TemporaryDirectory() as tmp:  # convert.py takes a folder, so give it only the new PDFs
            for p in todo:
                shutil.copy(p, Path(tmp) / p.name)
            if run([py, "convert.py", tmp, str(md_dir)], "convert") != 0:
                log.error("convert failed")
    for p in pdfs:
        if not raw(p.stem).exists() and not corrected(p.stem).exists():
            failed[p.stem] = "convert"
    active = [s for s in stems if s not in failed]

    # ---- stage 2: heading hierarchy
    log.info("===== STAGE 2/4: fix heading hierarchy")
    for s in active:
        if corrected(s).exists():
            log.info("skip headings %s: %s exists", s, corrected(s).name)
            remove(raw(s))  # stage 1 may have recreated the original next to an old corrected copy
            continue
        run([py, "headings_hierarchy.py", str(raw(s))], "headings")
        if not corrected(s).exists():
            failed[s] = "headings"
    active = [s for s in stems if s not in failed]

    # ---- stage 3: experience lines
    log.info("===== STAGE 3/4: find experience lines (embedding section gate)")
    for s in active:
        if experience(s).exists():
            log.info("skip experience %s: %s exists", s, experience(s).name)
            continue
        cmd = [py, "experience_lines.py", str(corrected(s)), "--threshold", str(args.threshold)] + gate_args
        if args.prefilter:
            cmd.append("--prefilter")
        if args.gate_headings:
            cmd.append("--gate-headings")
        run(cmd, "experience")
        if not experience(s).exists():
            failed[s] = "experience"
    active = [s for s in stems if s not in failed]

    # ---- stage 4: date intervals
    log.info("===== STAGE 4/4: extract date intervals")
    for s in active:
        if result(s).exists():
            log.info("skip intervals %s: %s exists", s, result(s).name)
            continue
        run([py, "extract_date_intervals.py", str(experience(s)), "--out-dir", str(out_dir)] + gate_args, "intervals")
        if not result(s).exists():
            failed[s] = "intervals"

    # ---- summary
    log.info("===== SUMMARY (%.1fs)", time.time() - t0)
    for s in stems:
        if s in failed:
            log.error("%-45s FAILED at stage: %s", s, failed[s])
        else:
            log.info("%-45s OK -> %s", s, result(s))
    log.info("log file: %s", log_path)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
