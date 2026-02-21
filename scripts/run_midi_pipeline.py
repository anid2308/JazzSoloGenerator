#!/usr/bin/env python3
"""
Test run: MIDI → tokens → (optional) round-trip MIDI.
Uses data/raw for input and data/processed for output.
Extracted from notebooks/JazzSoloGen_4.ipynb so you can run locally and see what happens.
"""
from collections import defaultdict
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")

# So we can import midi_tokenizer when run as python scripts/run_midi_pipeline.py
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)

from midi_tokenizer import (
    STEPS_PER_BEAT,
    clean_to_monophonic,
    get_vocab,
    midi_to_tokens,
    tokens_from_mono_with_pos,
    tokens_to_midi,
)

try:
    import miditoolkit
except ImportError:
    print("miditoolkit not installed. Run: pip install miditoolkit")
    sys.exit(1)


def max_distinct_pitches_same_start(m):
    inst = m.instruments[0]
    by = defaultdict(set)
    for n in inst.notes:
        by[n.start].add(n.pitch)
    return max((len(v) for v in by.values()), default=0)


def max_simultaneous_notes(notes):
    events = []
    for n in notes:
        events.append((n.start, 1))
        events.append((n.end, -1))
    events.sort()
    cur = mx = 0
    for _, d in events:
        cur += d
        mx = max(mx, cur)
    return mx


def gather_midi_paths():
    os.makedirs(RAW_DIR, exist_ok=True)
    paths = sorted(
        __import__("glob").glob(os.path.join(RAW_DIR, "**/*.mid"), recursive=True)
    ) + sorted(
        __import__("glob").glob(os.path.join(RAW_DIR, "**/*.midi"), recursive=True)
    )
    return paths


def create_test_midi():
    """Create one minimal test MIDI in data/raw so we can run the pipeline."""
    path = os.path.join(RAW_DIR, "test_snippet.mid")
    m = miditoolkit.MidiFile(ticks_per_beat=480)
    m.tempo_changes = [miditoolkit.TempoChange(120, 0)]
    inst = miditoolkit.Instrument(program=56, is_drum=False, name="Trumpet")
    # a few notes: start, end, pitch
    for start, end, pitch in [(0, 480, 60), (480, 960, 64), (960, 1440, 67)]:
        inst.notes.append(
            miditoolkit.Note(velocity=90, pitch=pitch, start=start, end=end)
        )
    m.instruments.append(inst)
    m.dump(path)
    return path


def main():
    print("=== MIDI → Tokens pipeline (test run) ===\n")

    midi_paths = gather_midi_paths()
    if not midi_paths:
        print("No MIDI files in data/raw. Creating a small test file...")
        test_path = create_test_midi()
        midi_paths = [test_path]
        print("Created:", test_path, "\n")

    print("MIDI files found:", len(midi_paths))
    for p in midi_paths[:5]:
        print("  -", os.path.basename(p))
    if len(midi_paths) > 5:
        print("  ... and", len(midi_paths) - 5, "more")

    # 1) Quick scan: stats per file
    print("\n--- Per-file stats (tracks, drums, notes, same_start, max_poly) ---")
    print("name                          | tracks | drums | notes | same_start | max_poly")
    for p in midi_paths:
        m = miditoolkit.MidiFile(p)
        inst = m.instruments[0]
        same = max_distinct_pitches_same_start(m)
        poly = max_simultaneous_notes(inst.notes)
        name = os.path.basename(p)[:28]
        print(f"{name:28} | {len(m.instruments):6d} | {str(any(i.is_drum for i in m.instruments)):5s} | {len(inst.notes):5d} | {same:10d} | {poly:8d}")

    # 2) Tokenize each file
    print("\n--- Tokenize (clean to mono → BAR/POS/NOTE/DUR/REST) ---")
    per_file_tokens = []
    for p in midi_paths:
        toks = midi_to_tokens(p)
        per_file_tokens.append(toks)

    total_tokens = sum(len(t) for t in per_file_tokens)
    print("Files:", len(per_file_tokens))
    print("Total tokens:", total_tokens)
    print("Example (first file, first 60 tokens):", per_file_tokens[0][:60])

    vocab, stoi, itos = get_vocab()
    print("\nVocab size:", len(vocab))

    # 3) Save first file's tokens as JSON and round-trip to MIDI
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    first_name = os.path.splitext(os.path.basename(midi_paths[0]))[0]

    tokens_path = os.path.join(PROCESSED_DIR, f"{first_name}_tokens.json")
    with open(tokens_path, "w") as f:
        json.dump(per_file_tokens[0], f, indent=0)
    print("\nWrote tokens JSON:", tokens_path)

    roundtrip_path = os.path.join(PROCESSED_DIR, f"{first_name}_roundtrip.mid")
    tokens_to_midi(per_file_tokens[0], out_path=roundtrip_path, tempo=120)
    print("Wrote round-trip MIDI:", roundtrip_path)

    print("\n=== Done. Check data/processed for tokens JSON and roundtrip MIDI. ===")


if __name__ == "__main__":
    main()
