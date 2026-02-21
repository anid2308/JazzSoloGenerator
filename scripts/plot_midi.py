#!/usr/bin/env python3
"""Plot MIDI files as score-style piano rolls. Requires: pip install pretty_midi matplotlib."""
import argparse
import os
import sys
from typing import List, Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pretty_midi
except ImportError as e:
    print("Missing dependency:", e, file=sys.stderr)
    print("Install with: pip install pretty_midi matplotlib", file=sys.stderr)
    sys.exit(1)

from paths import OUTPUTS_DIR, OUTPUTS_PLOTS_DIR

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_pitch_to_name(pitch: int) -> str:
    """e.g. 60 -> 'C4', 61 -> 'C#4'."""
    return NOTE_NAMES[pitch % 12] + str(pitch // 12 - 1)


def find_midis(directory: str) -> List[str]:
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return []
    midis = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith((".mid", ".midi")):
            midis.append(os.path.join(directory, name))
    return midis


def _get_tempo_bpm(pm: "pretty_midi.PrettyMIDI") -> float:
    """First tempo in BPM, or 120 if none."""
    times, tempos = pm.get_tempo_changes()
    if len(tempos) and tempos[0] > 0:
        return tempos[0]  # already in BPM in pretty_midi
    return 120.0


def plot_piano_roll(
    midi_path: str,
    ax=None,
    *,
    instrument_index: int = 0,
    show_velocity: bool = True,
    score_style: bool = True,
) -> "plt.Axes":
    """Plot one MIDI file as a piano roll (optionally score-style). Returns the axes used."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    if not pm.instruments:
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))
        ax.set_title(os.path.basename(midi_path) + " (no instruments)")
        return ax

    inst = pm.instruments[min(instrument_index, len(pm.instruments) - 1)]
    if not inst.notes:
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 4))
        ax.set_title(os.path.basename(midi_path) + " (no notes)")
        return ax

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    tempo = _get_tempo_bpm(pm)
    # Time in beats: beat = time_sec * (tempo / 60)
    sec_to_beat = tempo / 60.0

    y_min = min(n.pitch for n in inst.notes) - 1
    y_max = max(n.pitch for n in inst.notes) + 2

    if score_style:
        ax.set_facecolor("#faf8f5")
        ax.set_axisbelow(True)
        # Horizontal grid at every semitone (staff-like)
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.grid(True, axis="y", color="#e0dcd6", linewidth=0.8)
        ax.grid(True, axis="x", color="#e8e6e2", linewidth=0.5, alpha=0.8)
        # Bar lines every 4 beats
        end_beat = max(n.end * sec_to_beat for n in inst.notes)
        for b in range(0, int(end_beat) + 1, 4):
            ax.axvline(b, color="#ccc8c0", linewidth=0.6, zorder=0)
        # Y-axis as note names
        y_ticks = list(range(int(y_min), int(y_max) + 1))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([midi_pitch_to_name(p) for p in y_ticks], fontsize=9)
        ax.set_ylim(y_min - 0.5, y_max + 0.5)
    else:
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="x", alpha=0.3)

    for n in inst.notes:
        start_sec = n.start
        end_sec = n.end
        start_beat = start_sec * sec_to_beat
        dur_beat = (end_sec - start_sec) * sec_to_beat
        pitch = n.pitch
        if score_style:
            vel = n.velocity / 127.0 if show_velocity else 0.85
            # Dark note heads, slightly darker = stronger velocity
            gray = max(0.15, 0.5 - vel * 0.35)
            color = (gray, gray, gray, 1.0)
            ax.barh(pitch, dur_beat, left=start_beat, height=0.85, color=color, edgecolor="#333", linewidth=0.4)
        else:
            vel = n.velocity / 127.0 if show_velocity else 0.8
            color = plt.cm.viridis(vel)
            ax.barh(pitch, end_sec - start_sec, left=start_sec, height=0.8, color=color, edgecolor="none")

    if score_style:
        ax.set_xlabel("Beat")
        ax.set_ylabel("Pitch")
    else:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MIDI pitch")
    ax.set_title(f"{os.path.basename(midi_path)} — {inst.name or 'Instrument ' + str(instrument_index)}")
    return ax


def main():
    parser = argparse.ArgumentParser(
        description="Plot MIDI files as score-style piano rolls.",
        epilog="Which files: pass one .mid/.midi file to plot only that file; pass a directory to plot every .mid/.midi inside it (default: outputs/). Requires: pip install pretty_midi matplotlib",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=OUTPUTS_DIR,
        help="Single .mid/.midi file (plot that file only) or directory (plot all MIDI in it); default: outputs/",
    )
    parser.add_argument(
        "-o", "--out",
        metavar="FILE",
        help="Save figure to FILE (if no path, saves under outputs/plots/)",
    )
    parser.add_argument(
        "-i", "--instrument",
        type=int,
        default=0,
        help="Instrument index to plot (default: 0)",
    )
    parser.add_argument(
        "--no-velocity",
        action="store_true",
        help="Do not shade notes by velocity",
    )
    parser.add_argument(
        "--no-score-style",
        action="store_true",
        help="Use plain piano-roll style (time in sec, MIDI pitch numbers) instead of score style",
    )
    args = parser.parse_args()

    path = os.path.abspath(args.path)
    if os.path.isfile(path) and path.lower().endswith((".mid", ".midi")):
        midis = [path]
    elif os.path.isdir(path):
        midis = find_midis(path)
        if not midis:
            print(f"No .mid/.midi files in {path}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Not a file or directory: {path}", file=sys.stderr)
        sys.exit(1)

    score_style = not args.no_score_style
    n = len(midis)
    if n == 1:
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_piano_roll(
            midis[0],
            ax=ax,
            instrument_index=args.instrument,
            show_velocity=not args.no_velocity,
            score_style=score_style,
        )
    else:
        fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), squeeze=False)
        for ax, mp in zip(axes.ravel(), midis):
            plot_piano_roll(
                mp,
                ax=ax,
                instrument_index=args.instrument,
                show_velocity=not args.no_velocity,
                score_style=score_style,
            )
    plt.tight_layout()

    if args.out:
        out_path = args.out
        if not os.path.dirname(out_path):
            # Bare filename (e.g. all.png) → save under outputs/plots/
            os.makedirs(OUTPUTS_PLOTS_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUTS_PLOTS_DIR, os.path.basename(out_path))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print("Saved:", out_path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
