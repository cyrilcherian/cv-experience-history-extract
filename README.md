# cv-experience-history-extract

Extract work-experience date intervals from CV PDFs using only small, local models (Ollama + docling). Everything runs on a laptop; no API calls.

## Pipeline

`run_pipeline.py` runs four stages in order and logs each one:

| # | Script | Input | Output | Model |
|---|---|---|---|---|
| 1 | `convert.py` | `cvs/*.pdf` | `mds/<name>.md` | docling (no LLM) |
| 2 | `headings_hierarchy.py` | `mds/<name>.md` | `mds/<name>_corrected.md` | `qwen2.5:3b` |
| 3 | `experience_lines.py` | `<name>_corrected.md` | `<name>_corrected_experience.md` and `.log.jsonl` | `qwen2.5:1.5b` (logprobs) and `nomic-embed-text` (section gate) |
| 4 | `extract_date_intervals.py` | `<name>_corrected_experience.md` | `output/<name>_corrected_experience.json` | `qwen2.5:3b` |

- **Stage 2** fixes the heading levels (H1/H2/H3) that the PDF conversion gets wrong.
- **Stage 3** scores every line for "does it state years of experience or a from-to period". Only lines under sections whose heading is embedding-close to "work experience" are scored (`section_gate.py`). There is no keyword list.
- **Stage 4** sends each line to the model and writes the dates it returns, as is. There is no regex check or date parsing.

`section_gate.py` is the embedding gate used by stages 3 and 4. It can also be run on its own to calibrate the threshold.

## Folder structure

```
cv-experience-history-extract/
├── run_pipeline.py              # runs all four stages in order and logs them
├── convert.py                   # stage 1: PDF -> markdown
├── headings_hierarchy.py        # stage 2: fix heading levels
├── experience_lines.py          # stage 3: find experience lines
├── extract_date_intervals.py    # stage 4: model extracts the date intervals
├── section_gate.py              # embedding gate for stages 3 and 4
├── requirements.txt
├── cvs/                         # input PDFs (sample_cv.pdf is tracked; your own CVs are git-ignored)
├── mds/                         # generated markdown and stage 3 outputs
├── output/                      # final JSON per CV
├── logs/                        # one pipeline_<timestamp>.log per run
└── samples/
    └── sample_cv.html           # source of cvs/sample_cv.pdf
```

`mds/`, `output/` and `logs/` are created by the pipeline if missing, and their contents are git-ignored.

### Sample CV

`cvs/sample_cv.pdf` is a made-up CV for a fictional person ("Jane Doe"). All contact details are placeholders: `jane.doe@example.com`, `+00 000 000 0000`, and a fake address. It has work experience, education, projects and certifications sections, so you can try the pipeline right away:

```bash
python run_pipeline.py cvs/sample_cv.pdf
```

Do not commit real CVs. The `.gitignore` excludes everything in `cvs/` except the sample.

## Setup

### 1. Python environment

Python 3.10 or newer.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt     # docling (pulls in torch)
```

Only stage 1 needs Python packages. Stages 2 to 4 talk to Ollama over HTTP using the standard library.

### 2. Ollama and the models

Install Ollama from <https://ollama.com/download>, then start it and pull the three models:

```bash
ollama serve &                      # skip if it already runs as a service

ollama pull qwen2.5:3b              # stages 2 and 4
ollama pull qwen2.5:1.5b            # stage 3 line scoring
ollama pull nomic-embed-text        # stages 3 and 4 section gate (embeddings)

ollama list                         # confirm all three are listed
curl -s localhost:11434/api/tags    # confirm the server answers
```

`run_pipeline.py` checks that Ollama is reachable and that these models are pulled, and stops with a clear message if not. Ollama runs at `http://localhost:11434`.

## Run

Put PDFs in `cvs/`, then:

```bash
python run_pipeline.py                                  # every PDF in ./cvs
python run_pipeline.py "cvs/Jane Doe.pdf"               # one CV
python run_pipeline.py "cvs/Jane Doe.pdf" other_dir     # files and/or folders
python run_pipeline.py --force                          # redo all stages, replacing earlier outputs
python run_pipeline.py --prefilter                      # stage 3: only score lines with a digit or present/current word (faster)
python run_pipeline.py --threshold 0.6                  # stage 3 line threshold (default 0.50)
python run_pipeline.py --embed-threshold 0.7            # section gate (default 0.75)
python run_pipeline.py --embed-model bge-m3             # another embedding model (ollama pull it first)
python run_pipeline.py --no-section-gate                # score lines under every heading
python run_pipeline.py --gate-headings                  # extra stage 3 gate: ask the LLM per heading
python run_pipeline.py --md-dir mds --out-dir output --log-dir logs
```

A stage is skipped when its output already exists. Use `--force` to redo it.

Results are in `output/<name>_corrected_experience.json`:

```json
[
  {
    "line_no": 53,
    "context_path": ["Foo Bar", "EXPERIENCE / INTERNSHIP"],
    "raw_text": "- QBurst, Infopark Koratty | SEPT 2020 - Present Software Engineer",
    "start_date": "SEPT 2020",
    "end_date": "Present"
  }
]
```

Dates are exactly what the model returned, in document order.

### Logs

Every run writes `logs/pipeline_<timestamp>.log` and prints the same lines to the console. Each line is tagged with its stage (`[convert]`, `[headings]`, `[experience]`, `[intervals]`) and includes the command, the exit code and the time taken. A summary at the end lists OK or FAILED per CV. Stage 3 also writes every score to `mds/<name>_corrected_experience.log.jsonl`.

## Calibrating the section gate

The gate keeps a section if its heading's cosine similarity to the experience anchors reaches the threshold. To see the scores for a CV:

```bash
python section_gate.py "mds/Jane Doe_corrected.md"
python section_gate.py "mds/Jane Doe_corrected.md" --threshold 0.6
```

Headings marked `*` pass. If real experience sections score below the threshold, lower `--embed-threshold`. If everything falls below it, stage 3 scores no lines and the output is empty.

## Running the stages one at a time

```bash
python convert.py cvs mds
python headings_hierarchy.py "mds/Jane Doe.md"
python experience_lines.py "mds/Jane Doe_corrected.md" --prefilter
python extract_date_intervals.py "mds/Jane Doe_corrected_experience.md" --out-dir output
```

Always pass explicit file paths to stages 2 and 3. Their default is every `.md` in `mds/`, and running them without paths can feed earlier outputs back into the stage.

## Troubleshooting

- **`cannot reach Ollama`**: start it with `ollama serve`.
- **`model ... not found` or `HTTP 404`**: pull the missing model with `ollama pull <name>`.
- **Empty `output/*.json`**: check `mds/<name>_corrected_experience.log.jsonl` for the scores. Lines under the threshold are not kept. Lower `--threshold`, or `--embed-threshold` if the section gate dropped them.
- **Line missing from the output**: look for it in `mds/<name>_corrected_experience.md`. If it is there, stage 4's model returned no date for it.
