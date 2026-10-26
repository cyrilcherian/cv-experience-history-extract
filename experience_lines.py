#!/usr/bin/env python3
"""Find the lines of a markdown CV that very likely state work experience
(years of experience, or a from-to employment period) using a small
local Ollama model for decision making.

Every line is scored by default. With --gate-headings there are two stages:
  1. every heading (with its parent headings) is scored for whether its section
     could hold experience history; only headings at or above --heading-threshold
     (default 0.50) are opened up.
  2. each line under those headings is scored on its own, sent together with the
     headings it sits under (H1 > H2 > H3 ...) as context.
Lines scoring at or above the threshold (default 0.50) are written, grouped by hierarchy, to
"<name>_experience.md" next to the source file.

The model is qwen2.5:1.5b running on the local Ollama (/api/chat). It is asked a
Yes/No question and the probability is read from the logprobs of its first
token: P(yes) = p(Yes) / (p(Yes) + p(No)).

Every probability (headings and lines, passed or not) is also written to
"<name>_experience.log.jsonl", one JSON record per line.

Usage:
    python experience_lines.py "mds/Sandeep Kumar_corrected.md"   # one file
    python experience_lines.py                      # every .md in ./mds
    python experience_lines.py mds/a.md other_dir   # specific files / folders
    python experience_lines.py --threshold 0.95 --force
"""
import argparse
import json
import os
import re
import math
import sys
import urllib.error
import urllib.request

from section_gate import EMBED_MODEL, EMBED_THRESHOLD, EmbedError, SectionGate

HOST = "http://localhost:11434"
MODEL = "qwen2.5:1.5b"
THRESHOLD = 0.50  # date-range job lines score 0.6-0.8 on a 1.5B model; project and hackathon noise stays under 0.2
HEADING_THRESHOLD = 0.50  # stage 1 is only a gate, so it is looser than the line threshold
OUTPUT_SUFFIX = "_experience"
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mds")
MAX_LINE_CHARS = 1500

HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Cheap pre-filter (opt-in): a line can only state years / a period if it has a
# digit or a "present/current/till date" style word.
CANDIDATE_RE = re.compile(r"\d|\b(present|current|till date|to date|ongoing|now)\b", re.I)

# The few-shot examples matter: without them a 1.5B model under-scores clear
# lines (0.77 for "Jan 2015 - Mar 2019") and misses "Total experience: 12 years".
QUESTION = (
    "Does the Line mention how long the person has worked: a number of years of "
    "experience, or a from-to date range of a job?\n"
    "Examples:\n'Total experience: 12 years' -> Yes\n'Java Developer (2014-2017)' -> Yes\n"
    "'Infosys | 2018 - Present' -> Yes\n'Bachelor of Engineering, 2010' -> No\n"
    "'Good knowledge of Spring' -> No\n'Mobile No: +91-8310218303' -> No"
)

HEADING_QUESTION = (
    "Could the section under the heading after 'Heading:' contain the person's work "
    "experience history, such as years of experience or employment periods with from and "
    "to dates? Judge by the heading and the headings above it."
)


def in_passing_section(hierarchy, gate):
    """True if the line has no headings, or any heading above it is close to "experience"."""
    return not hierarchy or any(gate.passes(title) for _, title in hierarchy)


def iter_lines(lines):
    """Yield (line_no, hierarchy, text, is_heading) for every scoreable line.

    hierarchy is the list of (level, title) headings the line sits under; for a
    heading line it holds only its ancestors.
    """
    stack, in_fence = [], False
    for no, raw in enumerate(lines, 1):
        text = raw.strip()
        if FENCE_RE.match(raw):
            in_fence = not in_fence
            continue
        if not re.search(r"\w", text):  # blank, ---, ***, table rules...
            continue
        m = None if in_fence else HEADING_RE.match(raw)
        if m and m.group(2):
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            yield no, list(stack), title, True
            stack.append((level, title))
        else:
            yield no, list(stack), text, False


def build_context(hierarchy, text):
    parts = [f"H{level}: {title}" for level, title in hierarchy]
    return "Headings:\n" + ("\n".join(parts) if parts else "(none)") + f"\n\nLine: {text[:MAX_LINE_CHARS]}\n"


SYSTEM_PROMPT = "You are a strict classifier. Answer with exactly one word: Yes or No."


def score(model, context, question=QUESTION):
    """Return the model's probability (0-1) that the answer to the question is Yes."""
    body = json.dumps({
        "model": model,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 10,
        "options": {"temperature": 0, "num_predict": 1},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n{question}\nAnswer Yes or No."},
        ],
    }).encode()
    req = urllib.request.Request(f"{HOST}/api/chat", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            first = json.load(resp)["logprobs"][0]
    except urllib.error.HTTPError as e:
        sys.exit(f"Ollama returned HTTP {e.code}: {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        sys.exit(f"Cannot reach Ollama at {HOST}: {e.reason}")
    yes = no = 0.0
    for t in first["top_logprobs"]:
        word = t["token"].strip().lower()
        if word == "yes":
            yes += math.exp(t["logprob"])
        elif word == "no":
            no += math.exp(t["logprob"])
    return yes / (yes + no) if yes + no else 0.0


def collect_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, n) for n in os.listdir(p) if n.lower().endswith(".md"))
        else:
            files.append(p)
    return [f for f in files if not os.path.splitext(f)[0].endswith(OUTPUT_SUFFIX)]


def write_report(path, out_path, hits, threshold, total):
    groups = {}
    for h in hits:
        key = tuple(title for _, title in h["hierarchy"]) or ("(no heading)",)
        groups.setdefault(key, []).append(h)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Experience lines: {os.path.basename(path)}\n\n")
        f.write(f"Model: {MODEL} | threshold: {threshold:.2f} | {len(hits)} of {total} lines selected\n")
        for key, items in groups.items():
            f.write(f"\n## {' > '.join(key)}\n\n")
            for h in items:
                f.write(f"- [{h['prob']:.2f}] (line {h['line']}) {h['text']}\n")


def heading_path(hierarchy, title):
    return [t for _, t in hierarchy] + [title]


def select_headings(items, model, heading_threshold, log=None):
    """Stage 1: ask the model once per distinct heading path whether it may hold
    experience history. Returns the set of passing heading paths (tuples)."""
    cache, passing = {}, set()
    headings = [it for it in items if it[3]]
    for i, (no, hierarchy, title, _) in enumerate(headings, 1):
        path = tuple(heading_path(hierarchy, title))
        if path not in cache:
            ctx = "Heading:\n" + "\n".join(f"{'  ' * n}H{n + 1}: {t}" for n, t in enumerate(path)) + "\n"
            cache[path] = score(model, ctx, HEADING_QUESTION)
        prob = cache[path]
        if log is not None:
            log.append({"stage": "heading", "prob": round(prob, 4), "passed": prob >= heading_threshold,
                        "line": no, "path": list(path)})
        mark = "*" if prob >= heading_threshold else " "
        print(f"{mark} heading {i}/{len(headings)} {prob:.2f} {' > '.join(path)[:90]}", file=sys.stderr)
        if prob >= heading_threshold:
            passing.add(path)
    return passing


def under_passing(item, passing):
    """True if the line sits directly under a passing heading.

    Only the nearest heading counts: a passing ancestor (typically the H1 person
    name, which scores middling) must not open every section beneath it.
    """
    _, hierarchy, _, _ = item
    return bool(hierarchy) and tuple(t for _, t in hierarchy) in passing


def process(path, model, threshold, force, prefilter=False, heading_threshold=HEADING_THRESHOLD,
            gate_headings=False, gate=None):
    root, ext = os.path.splitext(path)
    out_path = root + OUTPUT_SUFFIX + ext
    if os.path.exists(out_path) and not force:
        print(f"skip {path}: {out_path} exists (use --force)", file=sys.stderr)
        return
    with open(path, encoding="utf-8") as f:
        all_items = list(iter_lines(f.read().splitlines()))
    total = len(all_items)

    log = []
    items = [it for it in all_items if not it[3]]
    if gate is not None:
        for title in dict.fromkeys(t for it in items for _, t in it[1]):
            print(f"{'*' if gate.passes(title) else ' '} section {gate.score(title):.3f} {title[:80]}", file=sys.stderr)
        kept = [it for it in items if in_passing_section(it[1], gate)]
        print(f"== section gate ({gate.model}, >= {gate.threshold:.2f}): kept {len(kept)} of {len(items)} lines",
              file=sys.stderr)
        items = kept
    if gate_headings:
        print(f"== {path}: stage 1, scoring headings", file=sys.stderr)
        passing = select_headings(all_items, model, heading_threshold, log)
        items = [it for it in items if under_passing(it, passing)]
        print(f"== stage 2: scoring {len(items)} of {total} lines under {len(passing)} headings", file=sys.stderr)
    else:
        print(f"== {path}: scoring {len(items)} lines", file=sys.stderr)
    if prefilter:
        items = [it for it in items if CANDIDATE_RE.search(it[2])]

    hits = []
    for i, (no, hierarchy, text, _) in enumerate(items, 1):
        prob = score(model, build_context(hierarchy, text))
        log.append({"stage": "line", "prob": round(prob, 4), "passed": prob >= threshold,
                    "line": no, "path": [t for _, t in hierarchy], "text": text})
        mark = "*" if prob >= threshold else " "
        print(f"{mark} {i}/{len(items)} {prob:.2f} {text[:70]}", file=sys.stderr)
        if prob >= threshold:
            hits.append({"line": no, "hierarchy": hierarchy, "text": text, "prob": prob})

    write_report(path, out_path, hits, threshold, total)
    log_path = root + OUTPUT_SUFFIX + ".log.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for rec in log:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"-> {out_path} ({len(hits)} lines), probabilities logged to {log_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=[DEFAULT_DIR], help="md files or folders (default: ./mds)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="minimum probability, 0-1 (default 0.50)")
    ap.add_argument("--gate-headings", action="store_true",
                    help="stage 1: only score lines under headings the model says may hold experience "
                         "(faster, but can drop lines such as '8+ years' under SUMMARY)")
    ap.add_argument("--heading-threshold", type=float, default=HEADING_THRESHOLD,
                    help="with --gate-headings: minimum probability for a heading to be opened up (default 0.50)")
    ap.add_argument("--prefilter", action="store_true",
                    help="only send lines containing a digit or present/current-style word to the model (much faster)")
    ap.add_argument("--embed-model", default=EMBED_MODEL, help="Ollama embedding model for the section gate")
    ap.add_argument("--embed-threshold", type=float, default=EMBED_THRESHOLD,
                    help="min cosine similarity of a heading to 'work experience' (default %(default)s)")
    ap.add_argument("--no-section-gate", action="store_true", help="score lines under every heading")
    ap.add_argument("--force", action="store_true", help="redo files whose _experience output exists")
    args = ap.parse_args()

    try:
        gate = None if args.no_section_gate else SectionGate(args.embed_model, args.embed_threshold)
    except EmbedError as e:
        sys.exit(str(e))
    for path in collect_files(args.paths):
        process(path, args.model, args.threshold, args.force, args.prefilter, args.heading_threshold,
                args.gate_headings, gate)


if __name__ == "__main__":
    main()
