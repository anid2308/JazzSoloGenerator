# JazzSoloGenerator

Generate jazz-sounding solo lines from MIDI data using a lightweight, end-to-end ML pipeline (preprocess ➝ train ➝ generate ➝ evaluate).

> Status: **WIP / actively iterating** — current generations can sound **too random/jumpy** with **awkward register leaps**. The roadmap below lists fixes.

**On Windows?** See [First-time setup (Windows)](#first-time-setup-windows) for clone-to-run steps. All `python scripts/...` commands below work in Command Prompt or PowerShell once the venv is activated.

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

## First-time setup (Windows)

Do this once after you **clone or pull** the repo on a Windows machine.

1. **Open a terminal**  
   Command Prompt (`cmd`) or PowerShell. Right-click the project folder → “Open in Terminal” (if available), or `cd` to the project folder.

2. **Check Python**  
   You need Python 3.10 or newer.
   ```bat
   python --version
   ```
   If that fails, install Python from [python.org](https://www.python.org/downloads/) and tick “Add Python to PATH”.

3. **Create and activate the virtual environment** (project folder = the one that contains `scripts`, `data`, `requirements.txt`):
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   ```
   Your prompt should show `(.venv)`.

4. **Install dependencies**
   ```bat
   pip install -r requirements.txt
   ```

5. **Optional — play generated MIDI later**  
   Install [FluidSynth](https://github.com/FluidSynth/fluidsynth): `choco install fluidsynth` (or [download](https://github.com/FluidSynth/fluidsynth/releases) and add to PATH).  
   Put a `.sf2` SoundFont in **`data\soundfonts\`** (e.g. [GeneralUser GS](https://schristiancollins.com/generaluser.php)).

After this, any time you open a new terminal to work on the project: `cd` to the project folder, run **`.venv\Scripts\activate`**, then use the commands in the rest of this README.

---

## Quickstart

### 1) Environment

**Recommended:** Python 3.10+ and a virtual environment.

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (Command Prompt or PowerShell):**
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Or with Conda (any OS):**
```bash
conda create -n jazzsolo python=3.10 -y
conda activate jazzsolo
pip install -r requirements.txt
```

Always **activate the venv first** in each new terminal, then run e.g. `python scripts/batch_parse.py`.

---

## Full workflow: training → generation → playback

End-to-end steps to train a model, generate a solo, and play it. **Project root** = the folder that contains `scripts`, `data`, `notebooks`, and `requirements.txt` (same on Windows and macOS).

### 1. Put MIDI in place

- Put your jazz solo / etude `.mid` or `.midi` files in the **`data/raw/`** folder (Windows: `data\raw\` is the same).
- Optional: from the project root with venv activated, run `python scripts/run_midi_pipeline.py` to tokenize and write a sample to `data/processed/`.

### 2. Train the model (notebook)

- Open **`notebooks/JazzSoloGen_4.ipynb`** in Jupyter (e.g. VS Code, Cursor, or `jupyter notebook`) or [Google Colab](https://colab.research.google.com) (GPU recommended).
- Run all cells; when prompted for data, upload a ZIP of MIDI files or point the notebook at `data/raw`.
- When training finishes, the notebook saves a checkpoint. Put **`best.pt`** in the project’s **`checkpoints/`** folder so the path is **`checkpoints/best.pt`** (same on Windows).

### 3. Generate a solo

From the **project root** (the folder that contains `scripts` and `data`), with your venv **activated**:

```bash
python scripts/generate_solo.py
```

(Works the same in Command Prompt or PowerShell on Windows.)

This creates **`outputs/solo_001.mid`**. To generate more or tweak sampling:

```bash
python scripts/generate_solo.py -n 3
python scripts/generate_solo.py --temperature 0.85 --top_k 15
```

### 4. Play the MIDI

Playback uses [FluidSynth](https://github.com/FluidSynth/fluidsynth) and a SoundFont (`.sf2`).

#### One-time setup

**Install FluidSynth**

- **macOS:** In Terminal, run: `brew install fluidsynth`
- **Windows:** In Command Prompt or PowerShell, run: `choco install fluidsynth`  
  Or [download](https://github.com/FluidSynth/fluidsynth/releases) the Windows build and add the folder containing `fluidsynth.exe` to your system PATH.

**SoundFont:** Put a `.sf2` file in the project folder **`data/soundfonts/`** (e.g. [GeneralUser GS](https://schristiancollins.com/generaluser.php) — download and place `GeneralUser-GS.sf2` in that folder). Or set the environment variable `FLUID_SOUNDFONT` to the full path of a `.sf2` file.

#### Running playback (be specific)

1. **Open a terminal** (macOS: Terminal; Windows: Command Prompt or PowerShell).
2. **Go to the project root** — the folder that contains `scripts`, `data`, `outputs`, etc.
   - **Windows example:** `cd C:\Users\YourName\JazzSoloGenerator`
   - **macOS example:** `cd "/Users/YourName/Personal Code Projects/JazzSoloGenerator"`
3. **Activate the virtual environment** (do this once per terminal session):
   - **Windows (Command Prompt or PowerShell):**  
     `.venv\Scripts\activate`  
     (You should see `(.venv)` in the prompt.)
   - **macOS / Linux:**  
     `source .venv/bin/activate`
4. **Run the play script.** Use exactly this command, with a space after `outputs` and nothing else on the line:
   - **Windows or macOS:**
     ```bash
     python scripts/play_midi.py outputs
     ```
   - Do **not** paste or type anything else after `outputs` on the same line (e.g. no `source` or path); that will cause “unrecognized arguments” and the script will not run.

**What you should see:** The script prints how many MIDI files it found, then for each file it prints `Playing: <filename>` and the FluidSynth command it runs. You should hear each file play in order.

- **macOS:** Paths in the printed command look like `/opt/homebrew/bin/fluidsynth` and `'.../GeneralUser-GS.sf2'`.
- **Windows:** Paths use backslashes, e.g. `C:\...\fluidsynth.exe` and `C:\...\outputs\gen_A_t0.75_k10.mid`.

Both show a “FluidSynth runtime version …” line; then the next file plays (or “Done.” when finished).

**Play other folders:** Use the same script with a different directory name (one word, no extra text on the line):

```bash
python scripts/play_midi.py data/processed
python scripts/play_midi.py data/raw
```

**macOS only — no sound:** If the script runs but you hear nothing, try running FluidSynth directly in the same terminal (same project root, venv activated):

```bash
fluidsynth -a coreaudio -ni data/soundfonts/GeneralUser-GS.sf2 outputs/gen_A_t0.75_k10.mid
```

If that plays sound, run `python scripts/play_midi.py outputs` again from that same terminal; it should then play as well.

---

## Project Layout

JazzSoloGenerator/
  data/
    raw/                # source MIDI files
    processed/          # JSON and round-trip MIDI from scripts
    soundfonts/         # optional .sf2 for play_midi.py (FluidSynth)
  checkpoints/          # saved models
  outputs/              # generated MIDI files
    plots/              # saved plot images (e.g. plot_midi.py -o file.png)
  notebooks/           # Jupyter notebooks (e.g. JazzSoloGen_4.ipynb)
  src/
    data/
    models/
    utils/
  scripts/
    paths.py            # shared RAW_DIR, PROCESSED_DIR, SOUNDFONTS_DIR
    parse.py            # single-file MIDI → JSON (edit INPUT_NAME)
    batch_parse.py      # batch MIDI → JSON (events + chords)
    midi_tokenizer.py   # MIDI ↔ token sequence (BAR/POS/NOTE/DUR/REST)
    run_midi_pipeline.py  # tokenize data/raw, write tokens + round-trip MIDI
    play_midi.py        # play .mid in a dir (FluidSynth; .sf2 in data/soundfonts/)
    plot_midi.py        # plot MIDI as piano roll (pretty_midi + matplotlib)
    generate_solo.py    # generate new solo MIDI from checkpoint (train in notebook first)
    corpus_example.py   # music21 corpus demo
  requirements.txt
  README.md

## Data & scripts reference

- **data/raw/** — Put your source MIDI files here.
- **data/processed/** — JSON and round-trip MIDI from `batch_parse.py` and `run_midi_pipeline.py`.
- **data/soundfonts/** — Put a `.sf2` file here for `play_midi.py` (or set `FLUID_SOUNDFONT`).

**Preprocessing (optional):**

- `python scripts/batch_parse.py` — All MIDI in `data/raw` → JSON (events + chords) in `data/processed`.
- `python scripts/run_midi_pipeline.py` — Tokenize MIDI, write tokens JSON and a round-trip MIDI.
- Single-file: set `INPUT_NAME` in `scripts/parse.py`, then run `python scripts/parse.py`.

**Generation:** `python scripts/generate_solo.py` (see Full workflow above for checkpoint requirement).  
**Playback:** `python scripts/play_midi.py [directory]` (see Full workflow for FluidSynth + SoundFont setup).

**Plot MIDI (piano roll):** `pip install pretty_midi matplotlib` then `python scripts/plot_midi.py [path]` (default: `outputs/`). Use `-o plot.png` to save (writes to `outputs/plots/` if you give only a filename).

## Known Issues

Lines sound too random/jumpy
Register feels wrong (unexpected extreme highs/lows)
Phrasing lacks direction (no “sentence-like” contour yet)

## Roadmap (high-impact fixes)

1. Constrain pitch/register (fast win)
hard clamp generation to an instrument range (e.g., trumpet range)
soft penalty for notes outside a learned “comfort band”
2. Penalize huge leaps
add an interval jump penalty at sampling time
or bake it into training with an auxiliary loss / feature
3. Add musical structure signals
bar/beat tokens
phrase boundary tokens
rhythmic quantization + swing-aware offsets (optional)
4. Better sampling defaults
top-p (nucleus) sampling + moderate temperature
repetition penalty
“stay-in-register” bias
5. Conditioning (future)
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