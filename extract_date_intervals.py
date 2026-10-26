#!/usr/bin/env python3
"""Extract employment/education date intervals from the "*_corrected_experience.md"
files produced by experience_lines.py, using qwen2.5:3b on a local Ollama.

Input lines are either section headers ("## A > B > C") or scored data lines
("- [0.97] (line 52) 04/2025 - Present, Client: Albertsons"). The CV parser often
turns a date range into a heading ("## A > WORK EXPERIENCE > 09/2022 - 04/2025,"),
so every header segment is a candidate too, once per distinct
heading path. Headings are read from the experience file and also from the
matching "<name>_corrected.md" source next to it, which catches dated headings
that have no selected line under them and gives their real line number (in the
experience file alone it is the first data line under the heading, or 0). Each candidate is
cleaned and sent to
Ollama with a JSON-schema constrained answer. Intervals are kept in document order
and written to output/<name>.json.

Usage:
    python extract_date_intervals.py                    # every *_corrected_experience.md in ./mds
    python extract_date_intervals.py mds/x_corrected_experience.md other_dir
    python extract_date_intervals.py --out-dir output --model qwen2.5:3b
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request

from section_gate import EMBED_MODEL, EMBED_THRESHOLD, EmbedError, SectionGate
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:3b"
INPUT_SUFFIX = "_corrected_experience.md"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IN_DIR = BASE_DIR / "mds"
DEFAULT_OUT_DIR = BASE_DIR / "output"
REQUEST_TIMEOUT_S = 120
RETRIES = 3

log = logging.getLogger("date_intervals")

HEADER_RE = re.compile(r"^#{1,6}\s+(?P<path>.+?)\s*$")
DATA_RE = re.compile(r"^-\s*\[(?P<score>[0-9.]+)\]\s*\(line\s+(?P<line>\d+)\)\s?(?P<text>.*)$")
BULLET_RE = re.compile(r"^\s*[-*•]\s+")

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "has_date": {"type": "boolean"},
        "start_date": {"type": ["string", "null"]},
        "end_date": {"type": ["string", "null"]},
    },
    "required": ["has_date", "start_date", "end_date"],
}

SYSTEM_PROMPT = """You extract one date interval from a single CV line.
Copy the dates EXACTLY as written in the line; never reformat or invent them.
- "start_date": the first date of the range.
- "end_date": the second date of the range. Keep words such as Present, Current or "till date" as written.
- If the line has only one date, put it in start_date and use null for end_date.
- If the line has no employment, education or experience date, answer has_date=false with nulls.
Examples:
"04/2025 - Present, Client: Albertsons" -> {"has_date": true, "start_date": "04/2025", "end_date": "Present"}
"09/2022 - 04/2025," -> {"has_date": true, "start_date": "09/2022", "end_date": "04/2025"}
"July 2017 - August 2019 in Zycus Infotech" -> {"has_date": true, "start_date": "July 2017", "end_date": "August 2019"}
"Worked at Acme from Sept' 09 to Sept' 12." -> {"has_date": true, "start_date": "Sept' 09", "end_date": "Sept' 12"}
"Graduated with 8.5 CGPA" -> {"has_date": false, "start_date": null, "end_date": null}"""


@dataclass(frozen=True)
class Candidate:
    line_no: int
    context_path: list[str]
    raw_text: str
    clean_text: str


@dataclass(frozen=True)
class Interval:
    line_no: int
    context_path: list[str]
    raw_text: str
    start_date: str
    end_date: Optional[str]


class OllamaError(RuntimeError):
    pass


def clean_text(text: str) -> str:
    """Strip bullets and markdown noise and collapse whitespace."""
    text = BULLET_RE.sub("", text.strip())
    text = re.sub(r"[*_]+", "", text).replace("|", " ")
    return re.sub(r"\s+", " ", text).strip()


SRC_HEADING_RE = re.compile(r"^\s{0,3}(?P<hashes>#{1,6})\s+(?P<title>.*?)\s*#*\s*$")


def heading_prefixes(path: list[str]) -> list[list[str]]:
    """Every prefix of a heading path, so each heading segment is a candidate on its own.

    '... > 09/2022 - 04/2025, > Achievements/Tasks' yields the '... > 09/2022 - 04/2025,'
    prefix even if that heading never appears on its own. The model decides which hold dates.
    """
    return [path[: i + 1] for i in range(len(path))]


def source_headings(source: str) -> list[tuple[int, list[str]]]:
    """(line_no, full path) for each heading of the original markdown CV."""
    stack: list[tuple[int, str]] = []
    out: list[tuple[int, list[str]]] = []
    in_fence = False
    for no, raw in enumerate(source.splitlines(), 1):
        if re.match(r"^\s*(```|~~~)", raw):
            in_fence = not in_fence
            continue
        m = None if in_fence else SRC_HEADING_RE.match(raw)
        if not m or not m["title"]:
            continue
        level = len(m["hashes"])
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, m["title"].strip()))
        out.append((no, [t for _, t in stack]))
    return out


def parse_candidates(markdown: str, source: Optional[str] = None, gate=None) -> list[Candidate]:
    """Every data line and heading segment; no regex pre-filter, the model decides what holds a date.

    A heading counts once per distinct path, from either the experience file or the
    optional original `source` markdown (whose line numbers win).
    """
    out: list[Candidate] = []
    seen: set[tuple[str, ...]] = set()
    pending: list[list[str]] = []  # headings awaiting a line number

    def add_heading(hpath: list[str], line_no: int) -> None:
        key = tuple(hpath)
        if key not in seen:
            seen.add(key)
            out.append(Candidate(line_no, list(hpath), hpath[-1], clean_text(hpath[-1])))

    def flush(line_no: int) -> None:
        for hpath in pending:
            add_heading(hpath, line_no)
        pending.clear()

    if source:
        for no, hpath in source_headings(source):
            for prefix in heading_prefixes(hpath)[-1:]:  # only the heading's own segment
                add_heading(prefix, no)

    path: list[str] = []
    for raw in markdown.splitlines():
        header = HEADER_RE.match(raw)
        if header:
            flush(0)
            path = [p.strip() for p in header["path"].split(">") if p.strip()]
            pending.extend(p for p in heading_prefixes(path) if tuple(p) not in seen)
            continue
        data = DATA_RE.match(raw.strip())
        if not data:
            continue
        flush(int(data["line"]))
        cleaned = clean_text(data["text"])
        if cleaned:
            out.append(Candidate(int(data["line"]), list(path), data["text"].strip(), cleaned))
    flush(0)
    if gate is None:
        return out
    kept = [c for c in out if not c.context_path or any(gate.passes(seg) for seg in c.context_path)]
    if len(kept) != len(out):
        log.info("section gate dropped %d of %d candidates (headings far from 'experience')",
                 len(out) - len(kept), len(out))
    return kept


def source_for(path: Path) -> Optional[str]:
    """Text of the '<name>_corrected.md' that an experience file was derived from."""
    src = path.with_name(path.name.replace("_experience", "", 1))
    try:
        return src.read_text(encoding="utf-8") if src != path and src.is_file() else None
    except (OSError, UnicodeDecodeError) as e:
        log.warning("cannot read source %s: %s", src, e)
        return None


def call_ollama(model: str, text: str) -> dict[str, Any]:
    """One structured-output request, with retries on transient failures."""
    body = json.dumps({
        "model": model,
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Line: {text}"},
        ],
    }).encode()
    last: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                content = json.load(resp)["message"]["content"]
            data = json.loads(content)
            if not isinstance(data, dict) or not {"has_date", "start_date", "end_date"} <= data.keys():
                raise ValueError(f"response does not match schema: {content[:120]}")
            return data
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):  # bad model name / request: retrying will not help
                raise OllamaError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError, KeyError) as e:
            last = e
        log.warning("attempt %d/%d failed: %s", attempt, RETRIES, last)
        time.sleep(min(2 ** attempt, 8))
    raise OllamaError(f"giving up after {RETRIES} attempts: {last}")


def to_interval(cand: Candidate, answer: dict[str, Any]) -> Optional[Interval]:
    """Take the model's answer as is: no regex validation, no date parsing."""
    if not answer.get("has_date"):
        return None
    start, end = answer.get("start_date"), answer.get("end_date")
    if not isinstance(start, str) or not start.strip():
        return None
    return Interval(cand.line_no, cand.context_path, cand.raw_text, start.strip(),
                    end.strip() if isinstance(end, str) and end.strip() else None)


def process_file(path: Path, out_dir: Path, model: str, gate=None) -> Path:
    candidates = parse_candidates(path.read_text(encoding="utf-8"), source_for(path), gate)
    log.info("%s: %d candidate lines with a date token", path.name, len(candidates))
    intervals: list[Interval] = []
    for i, cand in enumerate(candidates, 1):
        try:
            answer = call_ollama(model, cand.clean_text)
        except OllamaError as e:
            log.error("line %d skipped: %s", cand.line_no, e)
            continue
        iv = to_interval(cand, answer)
        log.info("[%d/%d] line %d: %s", i, len(candidates), cand.line_no,
                 f"{iv.start_date} -> {iv.end_date}" if iv else "no date")
        if iv:
            intervals.append(iv)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (path.name.removesuffix(".md") + ".json")
    payload = [{
        "line_no": iv.line_no,
        "context_path": iv.context_path,
        "raw_text": iv.raw_text,
        "start_date": iv.start_date,
        "end_date": iv.end_date,
    } for iv in intervals]
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("-> %s (%d intervals)", out_path, len(payload))
    return out_path


def collect_inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(p.glob(f"*{INPUT_SUFFIX}"))
        elif p.is_file():
            files.append(p)
        else:
            log.error("not found: %s", p)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_IN_DIR],
                    help=f"files or folders (default: ./mds, matching *{INPUT_SUFFIX})")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--embed-model", default=EMBED_MODEL, help="Ollama embedding model for the section gate")
    ap.add_argument("--embed-threshold", type=float, default=EMBED_THRESHOLD)
    ap.add_argument("--no-section-gate", action="store_true", help="keep lines under every heading")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)

    try:
        gate = None if args.no_section_gate else SectionGate(args.embed_model, args.embed_threshold)
    except EmbedError as e:
        log.error("%s", e)
        return 1
    files = collect_inputs(args.paths)
    if not files:
        log.error("no input files found")
        return 1
    status = 0
    for f in files:
        try:
            process_file(f, args.out_dir, args.model, gate)
        except (OSError, UnicodeDecodeError) as e:
            log.error("%s: %s", f, e)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
