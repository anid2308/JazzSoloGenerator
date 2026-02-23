"""
Chord extraction from MIDI and chord-symbol → allowed pitches for chord-guided generation.

Public API:
  get_chords_from_midi(midi_path) → [(bar_index, chord_figure), ...]
  allowed_midi_pitches_for_chord(chord_figure, pitch_min, pitch_max, include_scale=True) → set of MIDI pitches

Used by generate_solo.py when --chords-midi is provided. Supports:
  - music21 chord symbols (notation-style MIDI)
  - miditoolkit markers (e.g. files from generate_chord_midi.py)
  - inferred chords from played notes (any backing MIDI with chord notes per bar).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Set, Tuple

# Pitch class (0-11) to root name for chord label
_PC_TO_ROOT = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Optional: music21 for reading chord symbols from MIDI (and for chord → pitches)
try:
    from music21 import converter, harmony
    _HAS_MUSIC21 = True
except ImportError:
    _HAS_MUSIC21 = False

# Optional: miditoolkit for reading chord markers (e.g. from generate_chord_midi.py)
try:
    import miditoolkit
    _HAS_MIDITOOLKIT = True
except ImportError:
    _HAS_MIDITOOLKIT = False

# Steps per bar (must match midi_tokenizer)
STEPS_PER_BEAT = 12
BEATS_PER_BAR = 4
BAR_STEPS = BEATS_PER_BAR * STEPS_PER_BEAT


def _chords_from_music21(midi_path: str) -> List[Tuple[int, str]]:
    """Extract chord symbols via music21 (notation-style MIDI). Returns [(bar, figure), ...]."""
    try:
        score = converter.parse(midi_path)
        chord_symbols = list(score.flat.getElementsByClass(harmony.ChordSymbol))
    except Exception:
        return []
    if not chord_symbols:
        return []
    result = []
    for c in chord_symbols:
        offset = float(c.offset)
        bar = int(offset // BEATS_PER_BAR)
        fig = (c.figure or "").strip()
        if fig:
            result.append((bar, _normalize_chord_figure(fig)))
    by_bar = {}
    for bar, fig in result:
        by_bar[bar] = fig
    return sorted(by_bar.items())


def _chords_from_markers(midi_path: str) -> List[Tuple[int, str]]:
    """Extract chord symbols from MIDI markers (e.g. files from generate_chord_midi.py)."""
    if not _HAS_MIDITOOLKIT:
        return []
    try:
        m = miditoolkit.MidiFile(midi_path)
        tpb = m.ticks_per_beat
        ticks_per_bar = tpb * BEATS_PER_BAR
    except Exception:
        return []
    if not getattr(m, "markers", None):
        return []
    result = []
    for mk in m.markers:
        bar = int(mk.time // ticks_per_bar)
        fig = (getattr(mk, "text", None) or "").strip()
        if fig:
            result.append((bar, _normalize_chord_figure(fig)))
    by_bar = {}
    for bar, fig in result:
        by_bar[bar] = fig
    return sorted(by_bar.items())


# Chord templates: (quality_name, set of pitch classes relative to root 0)
_CHORD_TEMPLATES: List[Tuple[str, Set[int]]] = [
    ("m7", {0, 3, 7, 10}),
    ("7", {0, 4, 7, 10}),
    ("maj7", {0, 4, 7, 11}),
    ("m", {0, 3, 7}),
    ("", {0, 4, 7}),  # major
    ("dim", {0, 3, 6}),
    ("aug", {0, 4, 8}),
]


def _pitch_classes_to_chord(observed: Set[int]) -> str:
    """
    Infer a chord label from a set of pitch classes (0-11). Uses bass as root hint,
    then picks the best-matching template. Returns e.g. 'C', 'Cm7', 'F7'.
    """
    if not observed:
        return "C"  # fallback
    root_pc = min(observed)
    rotated = {(p - root_pc) % 12 for p in observed}
    best_name = ""
    best_score = -1
    for quality, template in _CHORD_TEMPLATES:
        overlap = len(template & rotated)
        # Prefer template that matches well (most of template present) and is most specific
        if overlap >= 2 and overlap >= len(template) * 0.5:
            score = overlap * 10 + len(template)  # prefer more specific (longer) on tie
            if score > best_score:
                best_score = score
                best_name = quality
    root_name = _PC_TO_ROOT[root_pc]
    return root_name + best_name if best_name else root_name


def _chords_from_notes(midi_path: str) -> List[Tuple[int, str]]:
    """
    Infer chord per bar from note events (e.g. piano comping / backing MIDI).
    Uses first non-drum instrument; groups notes by bar; maps pitch classes to chord label.
    """
    if not _HAS_MIDITOOLKIT:
        return []
    try:
        m = miditoolkit.MidiFile(midi_path)
        tpb = m.ticks_per_beat
        ticks_per_bar = tpb * BEATS_PER_BAR
    except Exception:
        return []
    # First instrument with notes (skip drums)
    inst = None
    for i in m.instruments:
        if getattr(i, "is_drum", False) or not getattr(i, "notes", None):
            continue
        if i.notes:
            inst = i
            break
    if not inst or not inst.notes:
        return []
    # Group pitch classes by bar (note overlaps bar if it sounds during that bar)
    by_bar: dict = defaultdict(set)
    for n in inst.notes:
        start_tick = int(n.start)
        end_tick = int(n.end)
        bar_start = (start_tick // ticks_per_bar) * ticks_per_bar
        bar_i = start_tick // ticks_per_bar
        # Include note in every bar it overlaps
        while bar_i * ticks_per_bar < end_tick:
            by_bar[bar_i].add(n.pitch % 12)
            bar_i += 1
    if not by_bar:
        return []
    result = []
    for bar in sorted(by_bar.keys()):
        pcs = by_bar[bar]
        label = _pitch_classes_to_chord(pcs)
        result.append((bar, _normalize_chord_figure(label)))
    return result


def get_chords_from_midi(midi_path: str) -> List[Tuple[int, str]]:
    """
    Extract chord symbols from a MIDI file. Returns a list of (bar_index, chord_figure).
    Tries in order: music21 chord symbols, then markers, then chord-from-notes inference
    (so any backing MIDI with played chords works). Bar index is 0-based. Returns [] if none found.
    """
    if _HAS_MUSIC21:
        result = _chords_from_music21(midi_path)
        if result:
            return result
    result = _chords_from_markers(midi_path)
    if result:
        return result
    return _chords_from_notes(midi_path)


def _normalize_chord_figure(fig: str) -> str:
    """Normalize chord symbol for lookup (e.g. 'C-7' -> 'Cm7', 'Cmin7' -> 'Cm7')."""
    fig = fig.strip()
    fig = re.sub(r"^([A-Ga-g]#?)([-−])", r"\1m", fig)  # C- -> Cm
    fig = re.sub(r"min(?!or)", "m", fig, flags=re.I)  # min7 -> m7
    fig = re.sub(r"maj", "M", fig, flags=re.I)  # maj7 -> M7
    return fig


def chord_to_pitch_classes(chord_figure: str) -> List[int]:
    """
    Return list of pitch classes (0-11) that are chord tones for the given symbol.
    Uses music21 if available; otherwise a small built-in table for common jazz symbols.
    """
    if _HAS_MUSIC21:
        try:
            cs = harmony.ChordSymbol(chord_figure)
            ch = cs.getChord()
            return sorted(set(p.pitchClass for p in ch.pitches))
        except Exception:
            pass
    return _chord_to_pitch_classes_fallback(chord_figure)


# Root name -> pitch class
_ROOT_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ROOT_PC["C#"] = 1
_ROOT_PC["Db"] = 1
_ROOT_PC["D#"] = 3
_ROOT_PC["Eb"] = 3
_ROOT_PC["F#"] = 6
_ROOT_PC["Gb"] = 6
_ROOT_PC["G#"] = 7
_ROOT_PC["Ab"] = 8
_ROOT_PC["A#"] = 10
_ROOT_PC["Bb"] = 10


def _chord_to_pitch_classes_fallback(fig: str) -> List[int]:
    """Fallback when music21 is not available: common jazz chord types."""
    if not fig:
        return list(range(12))
    fig = _normalize_chord_figure(fig)
    # Parse root (one or two chars)
    root_pc = None
    rest = fig
    for r in ("C#", "Db", "D#", "Eb", "F#", "Gb", "G#", "Ab", "A#", "Bb", "C", "D", "E", "F", "G", "A", "B"):
        if rest.startswith(r):
            root_pc = _ROOT_PC[r]
            rest = rest[len(r):].lstrip()
            break
    if root_pc is None:
        return list(range(12))  # allow all if unknown
    # Chord type -> intervals in semitones from root
    if not rest or rest in ("", "5"):
        return [root_pc, (root_pc + 7) % 12]  # power / fifth
    if rest.startswith("m") and "7" in rest:
        # m7, m9, etc.
        return [(root_pc + i) % 12 for i in (0, 3, 7, 10)]  # m7
    if rest.startswith("m"):
        return [(root_pc + i) % 12 for i in (0, 3, 7)]
    if "7" in rest and "M" not in rest and "maj" not in rest.lower():
        return [(root_pc + i) % 12 for i in (0, 4, 7, 10)]  # dom7
    if "M7" in rest or "maj7" in rest.lower() or rest == "M7":
        return [(root_pc + i) % 12 for i in (0, 4, 7, 11)]
    if rest.startswith("dim"):
        return [(root_pc + i) % 12 for i in (0, 3, 6, 9)]
    if rest.startswith("aug"):
        return [(root_pc + i) % 12 for i in (0, 4, 8)]
    # major triad default
    return [(root_pc + i) % 12 for i in (0, 4, 7)]


def allowed_midi_pitches_for_chord(
    chord_figure: str,
    pitch_min: int = 54,
    pitch_max: int = 84,
    include_scale: bool = True,
) -> set:
    """
    Return set of allowed MIDI pitches (in [pitch_min, pitch_max]) for the chord.
    If include_scale is True, add diatonic scale tones for the chord's root/quality (simple heuristic).
    """
    pcs = set(chord_to_pitch_classes(chord_figure))
    if include_scale and pcs:
        # Add common scale extensions (9, 13)
        root = min(pcs)  # use lowest pc as root
        for interval in (2, 6):
            pcs.add((root + interval) % 12)
    allowed = set()
    for pc in pcs:
        # All octaves in range that match this pitch class
        for p in range(pitch_min, pitch_max + 1):
            if p % 12 == pc:
                allowed.add(p)
    return allowed if allowed else set(range(pitch_min, pitch_max + 1))
