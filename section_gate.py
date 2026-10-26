#!/usr/bin/env python3
"""Embedding gate: is a CV section heading semantically close to "work experience"?

Each heading title is embedded with a local Ollama embedding model and compared (cosine
similarity) with a few anchor phrases about work history. The best similarity is the
heading's score; sections at or above the threshold are kept. There is no keyword list:
"EDUCATION" is dropped because it is far from the anchors, not because of its name.

Calibrate the threshold on your own CVs by printing the scores:
    python section_gate.py mds/x_corrected.md
    python section_gate.py mds/x_corrected.md --model bge-m3 --threshold 0.6
Pull a model first, for example:  ollama pull nomic-embed-text
"""
import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.request

HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_THRESHOLD = 0.75  # min similarity to the experience anchors; calibrate with `python section_gate.py <file.md>`

ANCHORS = [
    "work experience",
    "professional experience",
    "employment history",
    "career history and job positions",
    "internship and work history",
    "professional summary and total years of experience",
]

HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")


class EmbedError(RuntimeError):
    pass


def embed(texts, model=EMBED_MODEL):
    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(f"{HOST}/api/embed", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp)["embeddings"]
    except urllib.error.HTTPError as e:
        raise EmbedError(f"Ollama /api/embed HTTP {e.code} for model {model!r} "
                         f"(ollama pull {model}?): {e.read().decode()[:200]}")
    except urllib.error.URLError as e:
        raise EmbedError(f"cannot reach Ollama at {HOST}: {e.reason}")


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class SectionGate:
    """score(title) -> best cosine similarity to the experience anchors (cached)."""

    def __init__(self, model=EMBED_MODEL, threshold=EMBED_THRESHOLD):
        self.model, self.threshold = model, threshold
        # nomic models are trained with a task prefix; others ignore it harmlessly
        self.prefix = "classification: " if "nomic" in model else ""
        self.anchors = embed([self.prefix + a for a in ANCHORS], model)
        self.cache = {}

    def score(self, title):
        key = re.sub(r"\s+", " ", title).strip().lower()
        if key not in self.cache:
            vec = embed([self.prefix + key], self.model)[0]
            self.cache[key] = max(_cos(vec, a) for a in self.anchors)
        return self.cache[key]

    def passes(self, title):
        return self.score(title) >= self.threshold


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="markdown file whose headings to score")
    ap.add_argument("--model", default=EMBED_MODEL)
    ap.add_argument("--threshold", type=float, default=EMBED_THRESHOLD)
    args = ap.parse_args()
    try:
        gate = SectionGate(args.model, args.threshold)
        with open(args.file, encoding="utf-8") as f:
            titles = [m.group(2) for m in map(HEADING_RE.match, f.read().splitlines()) if m and m.group(2)]
        for t in titles:
            s = gate.score(t)
            print(f"{'*' if s >= gate.threshold else ' '} {s:.3f}  {t}")
    except EmbedError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
