#!/usr/bin/env python3
"""
Generate chord-progression MIDI files for testing chord-conditioned generation.

Writes 3 files to data/raw/chord_progressions/ (see paths.CHORD_PROGRESSIONS_DIR):
  01_diatonic_simple_C_F_G_Am.mid   — C, F, G, Am (diatonic, simple)
  02_slightly_complex_ii_V_I.mid    — Dm7, G7, Cmaj7, … (ii–V–I)
  03_more_complex_secondary_dominants.mid — longer progression with secondary dominants

Each chord is stored as a MIDI marker at the bar start. chord_utils.get_chords_from_midi()
reads these (or music21 chord symbols). Use with: generate_solo.py --chords-midi <path>.
"""
import os
import sys

from paths import CHORD_PROGRESSIONS_DIR

try:
    import miditoolkit
except ImportError:
    print("miditoolkit is required. pip install miditoolkit", file=sys.stderr)
    sys.exit(1)

from chord_utils import chord_to_pitch_classes

TICKS_PER_BEAT = 480
BEATS_PER_BAR = 4
TICKS_PER_BAR = TICKS_PER_BEAT * BEATS_PER_BAR
TEMPO = 100
# Piano-style voicing: root in this octave, others above (so backing is audible)
VOICING_ROOT_OCT = 3   # MIDI 48 = C3
VOICING_CHORD_OCT = 4  # MIDI 60 = C4


def _chord_to_voicing(chord_symbol: str) -> list:
    """Return list of MIDI pitches for a simple block chord (so the file is playable)."""
    pcs = chord_to_pitch_classes(chord_symbol)
    if not pcs:
        return [60, 64, 67]
    root_pc = min(pcs)
    root_midi = (VOICING_ROOT_OCT + 1) * 12 + root_pc
    others_pc = [p for p in pcs if p != root_pc]
    others = [(VOICING_CHORD_OCT + 1) * 12 + pc for pc in others_pc]
    return [root_midi] + others


def write_chord_midi(chord_sequence, out_path, add_notes=True):
    """
    Write a MIDI file with one marker per bar and optional chord notes so it's playable.
    chord_sequence: list of chord symbol strings (one per bar).
    add_notes: if True, add actual chord tones per bar so the file sounds when played.
    """
    m = miditoolkit.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    m.tempo_changes.append(miditoolkit.TempoChange(TEMPO, 0))
    for bar_i, chord in enumerate(chord_sequence):
        m.markers.append(miditoolkit.Marker(text=chord, time=bar_i * TICKS_PER_BAR))
    inst = miditoolkit.Instrument(program=0, is_drum=False, name="Chord reference")
    if add_notes:
        for bar_i, chord in enumerate(chord_sequence):
            start = bar_i * TICKS_PER_BAR
            end = start + TICKS_PER_BAR - 1  # hold chord for the bar
            for pitch in _chord_to_voicing(chord):
                inst.notes.append(miditoolkit.Note(velocity=80, pitch=pitch, start=start, end=end))
    m.instruments.append(inst)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    m.dump(out_path)
    return out_path


def main():
    os.makedirs(CHORD_PROGRESSIONS_DIR, exist_ok=True)

    # 1) Diatonic simple: C, F, G, Am (marked as such in filename)
    diatonic_simple = ["C", "F", "G", "Am"] * 4  # 16 bars
    path1 = os.path.join(CHORD_PROGRESSIONS_DIR, "01_diatonic_simple_C_F_G_Am.mid")
    write_chord_midi(diatonic_simple, path1)
    print("Wrote (diatonic, simple):", path1)

    # 2) Slightly more complex: ii–V–I and seventh chords
    slightly_complex = [
        "Dm7", "G7", "Cmaj7", "Cmaj7",
        "Em7", "A7", "Dmaj7", "Dmaj7",
        "Dm7", "G7", "Cmaj7", "Am7",
        "Dm7", "G7", "Cmaj7", "Cmaj7",
    ]
    path2 = os.path.join(CHORD_PROGRESSIONS_DIR, "02_slightly_complex_ii_V_I.mid")
    write_chord_midi(slightly_complex, path2)
    print("Wrote (slightly complex, ii–V–I):", path2)

    # 3) More complex: secondary dominants, altered, longer progression
    more_complex = [
        "Cm7", "F7", "Bbmaj7", "Bbmaj7",
        "Ebmaj7", "A7", "Dm7", "Dm7",
        "Gm7", "C7", "Fmaj7", "Fmaj7",
        "Am7", "D7", "Gmaj7", "Gmaj7",
        "Cm7", "F7", "Bbmaj7", "Ebmaj7",
        "Am7", "D7", "Gm7", "G7",
        "Cmaj7", "Cmaj7", "Fm7", "Bb7",
        "Ebmaj7", "Ebmaj7", "Cm7", "F7",
    ]
    path3 = os.path.join(CHORD_PROGRESSIONS_DIR, "03_more_complex_secondary_dominants.mid")
    write_chord_midi(more_complex, path3)
    print("Wrote (more complex, secondary dominants):", path3)

    print("\nAll chord progression MIDIs written to:", CHORD_PROGRESSIONS_DIR)
    print("Test with: python scripts/generate_solo.py --chords-midi", path1)


if __name__ == "__main__":
    main()
