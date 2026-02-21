# JazzSoloGenerator 🎷🤖
Generate jazz-sounding solo lines from MIDI data using a lightweight, end-to-end ML pipeline (preprocess ➝ train ➝ generate ➝ evaluate).

> Status: **WIP / actively iterating** — current generations can sound **too random/jumpy** with **awkward register leaps**. The roadmap below lists fixes.

---

## What this project does
**JazzSoloGenerator** trains a sequence model on jazz etude/solo MIDI files and generates new melodic lines as MIDI output. It’s built to be:
- **MIDI-first** (easy to train on public etudes / transcriptions)
- **Reproducible** (scripts for preprocessing, training, and generation)
- **Hackable** (swap models, sampling strategies, constraints)

---

## Core features
- **MIDI ingestion & cleaning**
  - Handles multi-track MIDI
  - Filters drum tracks
  - Extracts note events (pitch, start, duration, velocity)
- **Dataset stats + sanity checks**
  - Notes per file, polyphony, repeated start times, pitch range, etc.
- **Model training**
  - Train a sequence model (PyTorch) on extracted note-event sequences
- **Generation**
  - Sample new sequences and export to `.mid`
- **Evaluation helpers (basic but useful)**
  - Interval jump distribution
  - Pitch-range coverage
  - Note density (notes per bar/time)

---

## Quickstart

### 1) Environment
**Recommended:** Python 3.10+ and a virtual environment.

```bash
# create env (choose one)
python -m venv .venv
source .venv/bin/activate

# or conda
conda create -n jazzsolo python=3.10 -y
conda activate jazzsolo

# Install deps (from project root)
pip install -r requirements.txt

# Run scripts with this env: activate first, then e.g. python scripts/batch_parse.py
# (Or use .venv/bin/python scripts/batch_parse.py)
```
## Project Layout
JazzSoloGenerator/
  data/
    raw/                # source MIDI files
    processed/          # JSON from scripts/batch_parse.py
  checkpoints/         # saved models
  outputs/             # generated MIDI files
  notebooks/           # Jupyter notebooks (e.g. JazzSoloGen_4.ipynb)
  src/
    data/
    models/
    utils/
  scripts/
    parse.py            # single-file parse → JSON
    batch_parse.py      # batch MIDI → JSON (events + chords)
    corpus_example.py   # music21 corpus demo
    # preprocess.py, train.py, generate.py, evaluate.py (TBD)
  requirements.txt
  README.md

## Data
data/raw/

## Preprocessing
Scripts resolve `data/raw` and `data/processed` from the project root, so you can run them from any directory:

```bash
# Batch: all MIDI in data/raw → JSON in data/processed
python scripts/batch_parse.py
```

Single-file: edit the input path in `scripts/parse.py` then run `python scripts/parse.py`.
```bash
## Training
python scripts/train.py \
  --data_dir data/processed \
  --out_dir checkpoints \
  --epochs 50 \
  --batch_size 32 \
  --device cuda
```
```bash
## Generation
python scripts/generate.py \
  --checkpoint checkpoints/best.pt \
  --out_mid outputs/solo_001.mid \
  --num_notes 256 \
  --temperature 0.9 \
  --top_k 40
```
```bash
## Evaluation
python scripts/evaluate.py \
  --midi_dir outputs \
  --report outputs/eval_report.json
```

## Known Issues
Lines sound too random/jumpy
Register feels wrong (unexpected extreme highs/lows)
Phrasing lacks direction (no “sentence-like” contour yet)


## Roadmap (high-impact fixes)
1) Constrain pitch/register (fast win)
hard clamp generation to an instrument range (e.g., trumpet range)
soft penalty for notes outside a learned “comfort band”

2) Penalize huge leaps
add an interval jump penalty at sampling time
or bake it into training with an auxiliary loss / feature

3) Add musical structure signals
bar/beat tokens
phrase boundary tokens
rhythmic quantization + swing-aware offsets (optional)

4) Better sampling defaults
top-p (nucleus) sampling + moderate temperature
repetition penalty
“stay-in-register” bias

5) Conditioning (future)
chord progression conditioning
style conditioning (bebop, ballad, modal, etc.)

## Repro Tips
Set random seeds in preprocessing, training, and generation.
Save config snapshots with each checkpoint.
Keep a small “golden” MIDI set to regression-test preprocessing and generation.

## All feedback and contribution welcome
This is a personal project, but PRs/issues are welcome:
bug reports (attach a minimal MIDI that reproduces the issue)
model improvements
evaluation metrics
better preprocessing (quantization, swing, segmentation)

## License
MIT
