#!/usr/bin/env python3
"""Play MIDI files in a directory using FluidSynth. Put a .sf2 in data/soundfonts/ or set FLUID_SOUNDFONT."""
import argparse
import glob
import os
import shlex
import shutil
import subprocess
import sys
from typing import List, Optional

from paths import PROCESSED_DIR, SOUNDFONTS_DIR


def _soundfont_candidates() -> List[str]:
    out = []
    if os.environ.get("FLUID_SOUNDFONT"):
        out.append(os.environ["FLUID_SOUNDFONT"])
    if os.path.isdir(SOUNDFONTS_DIR):
        out.extend(sorted(glob.glob(os.path.join(SOUNDFONTS_DIR, "*.sf2"))))
    # Common system paths (Unix only; Windows uses FLUID_SOUNDFONT or data/soundfonts/)
    if sys.platform != "win32":
        out.extend([
            "/usr/share/soundfonts/FluidR3_GM.sf2",
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        ])
    return out


def find_soundfont(override: Optional[str]) -> Optional[str]:
    if override and os.path.isfile(override):
        return override
    for path in _soundfont_candidates():
        if path and os.path.isfile(path):
            return path
    return None


def find_midis(directory: str) -> List[str]:
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return []
    midis = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith((".mid", ".midi")):
            midis.append(os.path.join(directory, name))
    return midis


def _fluidsynth_binary() -> str:
    """Prefer full path so the script definitely runs FluidSynth (not something else on PATH)."""
    return shutil.which("fluidsynth") or "fluidsynth"


def play_with_fluidsynth(midi_path: str, soundfont: str) -> bool:
    # On macOS: run the exact FluidSynth command that works in the terminal
    # (fluidsynth -a coreaudio -ni <soundfont> <midi>) via shell so CoreAudio works.
    binary = _fluidsynth_binary()
    if sys.platform == "darwin":
        # Match the working command: fluidsynth -a coreaudio -ni soundfont.sf2 file.mid
        cmd_parts = [binary, "-a", "coreaudio", "-ni", soundfont, midi_path]
        cmd = " ".join(shlex.quote(p) for p in cmd_parts)
        try:
            print(f"  $ {cmd}", flush=True)
            subprocess.run(cmd, check=True, shell=True, stdin=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False
    # Windows / Linux: run as list (no shell)
    try:
        subprocess.run(
            [binary, "-ni", "-q", soundfont, midi_path],
            check=True,
            stdin=subprocess.DEVNULL,
        )
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False


def get_midis_from_path(path: str) -> List[str]:
    """Return list of MIDI file paths: one if path is a .mid/.midi file, else all in directory."""
    path = os.path.abspath(os.path.expanduser(path.strip()))
    if os.path.isfile(path):
        if path.lower().endswith((".mid", ".midi")):
            return [path]
        return []
    if os.path.isdir(path):
        return find_midis(path)
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Play MIDI files in a directory (or a single file) using FluidSynth.",
        epilog="Install FluidSynth (macOS: brew install fluidsynth; Windows: choco install fluidsynth). Need a .sf2 in data/soundfonts/ or set FLUID_SOUNDFONT.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Directory or path to a .mid/.midi file. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "-s", "--soundfont",
        metavar="SF2",
        help="Path to SoundFont .sf2 file (default: FLUID_SOUNDFONT or data/soundfonts/GeneralUser.sf2)",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Only list MIDI files, do not play",
    )
    args = parser.parse_args()

    path = args.path
    if path is None:
        try:
            path = input("Path to directory or MIDI file: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            sys.exit(0)
        if not path:
            print("No path entered.", file=sys.stderr)
            sys.exit(1)

    soundfont = find_soundfont(args.soundfont)
    if not soundfont and not args.dry_run:
        print("No SoundFont found.", file=sys.stderr)
        print("  Set FLUID_SOUNDFONT to a .sf2 path, or put one in data/soundfonts/ (e.g. GeneralUser.sf2).", file=sys.stderr)
        print("  Free SoundFonts: https://github.com/FluidSynth/fluidsynth/wiki/SoundFont", file=sys.stderr)
        sys.exit(1)

    midis = get_midis_from_path(path)
    if not midis:
        print(f"No .mid/.midi file(s) at {os.path.abspath(path)}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(midis)} MIDI file(s)")
    if args.dry_run:
        for p in midis:
            print(" ", os.path.basename(p))
        return

    # Check fluidsynth is available (same binary we'll use for playback)
    try:
        subprocess.run(
            [_fluidsynth_binary(), "-v"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except FileNotFoundError:
        if sys.platform == "darwin":
            print("fluidsynth not found. Install with: brew install fluidsynth", file=sys.stderr)
        elif sys.platform == "win32":
            print("fluidsynth not found. Install with: choco install fluidsynth", file=sys.stderr)
            print("  Or download from https://github.com/FluidSynth/fluidsynth/releases and add to PATH.", file=sys.stderr)
        else:
            print("fluidsynth not found. Install FluidSynth and add it to PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("fluidsynth -v timed out.", file=sys.stderr)
        sys.exit(1)

    for path in midis:
        name = os.path.basename(path)
        print(f"Playing: {name}", flush=True)
        if not play_with_fluidsynth(path, soundfont):
            print(f"  (fluidsynth failed for {name})", file=sys.stderr)
    print("Done.")


if __name__ == "__main__":
    main()
