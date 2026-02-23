"""Project paths. All scripts use data/raw, data/processed, etc. from repo root."""
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
CHORD_PROGRESSIONS_DIR = os.path.join(RAW_DIR, "chord_progressions")  # chord MIDIs for --chords-midi
PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
SOUNDFONTS_DIR = os.path.join(_PROJECT_ROOT, "data", "soundfonts")
CHECKPOINTS_DIR = os.path.join(_PROJECT_ROOT, "checkpoints")
OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")
OUTPUTS_PLOTS_DIR = os.path.join(_PROJECT_ROOT, "outputs", "plots")
