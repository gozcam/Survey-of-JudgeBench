# Survey of JudgeBench

This repository contains the scripts and small modifications used to run survey experiments on **JudgeBench**, comparing different LLM judge paradigms across four categories:

- **Knowledge** (`mmlu-pro`)
- **Reasoning** (`livebench-reasoning`)
- **Math** (`livebench-math`)
- **Coding** (`livecodebench`)

Three judge paradigms are evaluated:

| Paradigm | Models |
|---|---|
| Prompted | GPT-4o-mini, GPT-4.1-mini, Gemini 2.5 Flash Lite |
| Fine-tuned critic | Skywork-Critic-Llama-3.1-8B |
| Reward model | Skywork-Reward-Llama-3.1-8B |
| Prompted (local) | Meta-Llama-3.1-8B-Instruct (via vLLM) |

---

## Repository structure

```text
repo/
├─ data/                          # pilot subsets (10-pair)
├─ outputs/                       # judged results per model
│  └─ analysis/                   # summary.txt, comparison.csv, failures_*.jsonl
├─ figures/                       # generated figures (PDF + PNG, one per figure)
├─ scripts/
│  ├─ rungpt4omini_pilot.py
│  ├─ rungpt4omini_full.py
│  ├─ rungeminiflashlite_pilot.py
│  ├─ rungeminiflashlite_full.py
│  ├─ rungpt41mini_pilot.py
│  ├─ rungpt41mini_full.py
│  ├─ runskyworkcritic_pilot.py
│  ├─ runskyworkcritic_full.py
│  ├─ runskyworkreward_pilot.py
│  ├─ runskyworkreward_full.py
│  ├─ runllama31_8b_pilot.py
│  ├─ runllama31_8b_full.py
│  ├─ analyze_outputs.py
│  └─ figure.py
├─ localsetups/
│  ├─ local_setup_llama31_8b_vllm.md
│  ├─ local_setup_skywork_critic_vllm.md
│  └─ local_setup_skywork_reward_cuda.md
└─ third_party/
   └─ judgebench/
      ├─ run_judge.py
      ├─ utils/
      └─ data/
         └─ dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl
```

---

## Setup

### 1. Create and activate a virtual environment

**Windows Command Prompt**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux / WSL**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If needed for Gemini support:
```bash
pip install google-generativeai
```

If you hit an OpenAI/httpx compatibility error:
```bash
pip install httpx==0.27.2
```

For Skywork Reward (runs via HuggingFace transformers on GPU):
```bash
pip install -r third_party/judgebench/requirements-cuda.txt
```

---

## API keys

Set the required key in your terminal session before running.

**OpenAI models (GPT-4o-mini, GPT-4.1-mini)**
```bash
export OPENAI_API_KEY="your_key_here"      # macOS/Linux/WSL
set OPENAI_API_KEY=your_key_here           # Windows CMD
$env:OPENAI_API_KEY="your_key_here"        # PowerShell
```

**Gemini Flash Lite**
```bash
export GEMINI_API_KEY="your_key_here"
```

Local models (Skywork Critic, Skywork Reward, Llama 3.1 8B) do not require an API key.

---

## Required dataset

The full dataset must be present at:

```text
third_party/judgebench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl
```

Pilot scripts create a 10-pair subset at:

```text
data/dataset=judgebench-pilot10,response_model=gpt-4o-2024-05-13.jsonl
```

---

## Running experiments

Each model has a pilot script (10-pair subset) and a full-run script. Run them from the repo root:

```bash
python scripts/rungpt4omini_pilot.py
python scripts/rungpt4omini_full.py

python scripts/rungeminiflashlite_pilot.py
python scripts/rungeminiflashlite_full.py

python scripts/rungpt41mini_pilot.py
python scripts/rungpt41mini_full.py

python scripts/runskyworkcritic_pilot.py   # requires vLLM server on localhost:8000
python scripts/runskyworkcritic_full.py

python scripts/runskyworkreward_pilot.py   # requires CUDA GPU + HuggingFace deps
python scripts/runskyworkreward_full.py

python scripts/runllama31_8b_pilot.py      # requires vLLM server on localhost:8000
python scripts/runllama31_8b_full.py
```

For detailed setup instructions for the three local models, see [`localsetups/`](localsetups/).

> **Note:** these scripts are wrappers around `third_party/judgebench/run_judge.py`. The same runs can be performed directly via the terminal command documented in the [JudgeBench README](third_party/judgebench/README.md). The wrapper scripts are provided here for simplicity and to document the exact flags and paths used for each model.

---

## Analyzing results

After full runs are complete, generate the analysis:

```bash
python scripts/analyze_outputs.py
```

This reads all full-run JSONL files from `outputs/` and writes to `outputs/analysis/`:

- `summary.txt` — per-model accuracy tables, failure breakdown, cross-model and paradigm comparisons
- `comparison.csv` — spreadsheet-friendly accuracy table
- `failures_<model>.jsonl` — failure records per model for qualitative inspection

---

## Generating figures

After full runs and analysis are complete, generate all report figures:

```bash
python scripts/figure.py
```

This reads full-run JSONL files from `outputs/` and writes one PDF and one PNG per figure to `figures/` at the repo root (created automatically if it doesn't exist).

| File | Description |
|---|---|
| `figure1_failure_modes` | Stacked bar: correct / incorrect / inconsistent per model |
| `figure2_category_accuracy` | Grouped bar: accuracy by category and model |
| `figure3_inconsistency_heatmap` | Heatmap: order-swap inconsistency rate by model and category |
| `figure4_agreement_matrix` | Heatmap: pairwise verdict agreement between prompted judges |
| `figure5_consensus` | Horizontal stacked bar: consensus distribution across prompted models |
| `figure6_score_margin` | Grouped bar: Skywork-Reward score margin for correct vs. incorrect pairs |
| `figure7_incorrect_heatmap` | Heatmap: incorrect judgment rate by model and category |

PDFs are suitable for LaTeX inclusion; PNGs are for preview and quick inspection.

---

## Output files

Judged results are written to `outputs/` with filenames like:

```text
dataset=judgebench,response_model=gpt-4o-2024-05-13,judge_name=arena_hard,judge_model=gpt-4o-mini.jsonl
```

Each JSONL row contains the original pair fields plus `judgments`. If an output file already exists, `run_judge.py` resumes from where it left off. Delete the file first if you want a clean rerun.

---

## Troubleshooting

**`OPENAI_API_KEY is not set`** — export the key in the same terminal session before running.

**`AsyncClient.__init__() got an unexpected keyword argument 'proxies'`** — pin `httpx==0.27.2`.

**Unicode / `cp1252` write errors on Windows** — output files must be opened with `encoding="utf-8"`.

**Pilot metrics crash on empty category** — a 10-pair subset may miss a category entirely; the metrics code guards against this, but very small subsets are expected to have sparse results.
