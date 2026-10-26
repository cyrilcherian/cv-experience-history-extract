#!/usr/bin/env python3
"""Derive a logical H1/H2/H3... hierarchy for the headings of markdown files
using a local Ollama model (qwen2.5:3b), classifying one heading at a time,
then rewrite each file with the corrected heading levels.

Each processed file is renamed to "<name>_corrected.md". Files already ending in
"_corrected" are skipped, so re-running the program does not adjust them again.

Usage:
    python headings_hierarchy.py                    # every .md in ./mds
    python headings_hierarchy.py mds/a.md other_dir # specific files / folders
    python headings_hierarchy.py --json-dir out     # also dump title/level JSON
"""
import argparse
import json
import os
import re
import sys
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
#MODEL = "qwen2.5:3b"
MODEL = "qwen2:7b"
MAX_LEVEL = 4
CORRECTED_SUFFIX = "_corrected"
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mds")

HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")

SYSTEM_PROMPT = """You organise the headings of a CV/resume into an outline. For each new heading, decide how it relates to the PREVIOUS heading:
- "child": the new heading is a part/detail INSIDE the previous heading (e.g. "Responsibilities" after "Project 1: Foo").
- "sibling": same level as the previous heading (e.g. "Project 2" after "Project 1"; "EDUCATION" after "SKILLS").
- "shallower": the previous heading's section is finished and the new heading is a bigger section, so it goes up one or more levels (e.g. "EDUCATION" after "Responsibilities").
Hints: top-level sections (SUMMARY, SKILLS, EXPERIENCE, PROJECT DETAILS, EDUCATION, TRAINING...) are siblings of each other and never children of each other. Only a project/job title or generic detail heading (Responsibilities, Environment) is a child.

Reply with JSON only: {"relation": "child" | "sibling" | "shallower"}"""


def find_headings(lines):
    """Return [(line_index, title)] in order, skipping fenced code blocks."""
    found, in_fence = [], False
    for i, line in enumerate(lines):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(line)
        if m and m.group(1):
            found.append((i, m.group(1).strip()))
    return found


def ask_level(all_headings, decided, heading, max_allowed):
    outline = "\n".join(f"{'  ' * (h['level'] - 1)}H{h['level']} {h['title']}" for h in decided) or "(empty)"
    user = (
        "All headings in order:\n"
        + "\n".join(f"- {h}" for h in all_headings)
        + f"\n\nOutline so far:\n{outline}\n\nPrevious heading: {decided[-1]["title"]}\nNew heading: {heading}"
    )
    body = json.dumps({
        "model": MODEL,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        content = json.load(resp)["message"]["content"]
    try:
        relation = json.loads(content)["relation"]
    except (ValueError, KeyError, TypeError):
        relation = "sibling"
    prev = decided[-1]["level"]
    if relation == "child":
        level = prev + 1
    elif relation == "shallower":
        # go back up to the nearest earlier heading that is a major section
        level = 2
    else:
        level = prev
    return min(max(level, 2), max_allowed)


def pattern_key(title):
    """'Product 2: BTS' -> 'product #' so numbered siblings can be matched."""
    head = re.split(r"[:\-(]", title, maxsplit=1)[0].lower()
    key = re.sub(r"\d+", "#", head).strip()
    return key if "#" in key else None


def build_hierarchy(headings):
    decided = []
    for i, title in enumerate(headings):
        max_allowed = 1 if i == 0 else min(decided[-1]["level"] + 1, MAX_LEVEL)
        key = pattern_key(title)
        twin = next((d for d in decided if key and pattern_key(d["title"]) == key), None)
        if i == 0:
            level = 1
        elif twin:  # numbered sibling of an earlier heading (Product 1 / Product 2)
            level = twin["level"]
        else:
            level = max(2, ask_level(headings, decided, title, max_allowed))
        level = min(level, max_allowed)
        decided.append({"title": title, "level": level})
        print(f"{'  ' * (level - 1)}H{level} {title}", file=sys.stderr)
    return [{"title": d["title"], "hierarchy": f"H{d['level']}"} for d in decided]


def collect_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, n) for n in os.listdir(p) if n.lower().endswith(".md"))
        else:
            files.append(p)
    return files


def is_corrected(path):
    return os.path.splitext(path)[0].endswith(CORRECTED_SUFFIX)


def correct_file(path, json_dir=None):
    root, ext = os.path.splitext(path)
    new_path = root + CORRECTED_SUFFIX + ext
    if os.path.exists(new_path):
        print(f"skip {path}: {new_path} already exists", file=sys.stderr)
        return
    with open(path, encoding="utf-8", newline="") as f:
        lines = f.read().splitlines(keepends=True)
    found = find_headings(lines)
    if not found:
        print(f"skip {path}: no headings found", file=sys.stderr)
        return

    print(f"== {path}", file=sys.stderr)
    result = build_hierarchy([title for _, title in found])

    for (i, title), item in zip(found, result):
        ending = lines[i][len(lines[i].rstrip("\r\n")):]
        lines[i] = f"{'#' * int(item['hierarchy'][1:])} {title}{ending}"

    # write the corrected copy first, then drop the original (rename semantics)
    with open(new_path, "w", encoding="utf-8", newline="") as f:
        f.writelines(lines)
    os.remove(path)
    print(f"-> {new_path}", file=sys.stderr)

    if json_dir:
        os.makedirs(json_dir, exist_ok=True)
        name = os.path.basename(root) + ".json"
        with open(os.path.join(json_dir, name), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=[DEFAULT_DIR], help="md files or folders (default: ./mds)")
    ap.add_argument("--json-dir", help="also write the title/hierarchy JSON per file into this folder")
    args = ap.parse_args()

    files = collect_files(args.paths)
    todo = [f for f in files if not is_corrected(f)]
    print(f"{len(todo)} to process, {len(files) - len(todo)} already corrected", file=sys.stderr)
    for path in todo:
        correct_file(path, args.json_dir)


if __name__ == "__main__":
    main()
