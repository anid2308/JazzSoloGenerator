"""
MIDI → token sequence (and back) for the jazz solo model.
Extracted from notebooks/JazzSoloGen_4.ipynb so it can be run as scripts.
Uses miditoolkit; no Colab or torch required for tokenize/round-trip.
"""
from collections import defaultdict
import os

try:
    import miditoolkit
except ImportError:
    miditoolkit = None

# Same as notebook
STEPS_PER_BEAT = 12
BAR_STEPS = 4 * STEPS_PER_BEAT
MAX_DUR = 48
MAX_REST = 48

PAD, BOS, EOS = "<PAD>", "<BOS>", "<EOS>"


def ticks_to_steps(ticks, tpb, spb=STEPS_PER_BEAT):
    return int(round((ticks / tpb) * spb))


def clean_to_monophonic(notes, tpb, spb=STEPS_PER_BEAT):
    """
    Convert messy/polyphonic notes into a single line by:
    1) quantize to steps
    2) keep 1 note per onset step (longest dur, tie -> highest pitch)
    3) trim overlaps
    Returns list of (start_step, end_step, pitch).
    """
    if not notes:
        return []

    q = []
    for n in notes:
        s = ticks_to_steps(n.start, tpb, spb)
        e = ticks_to_steps(n.end, tpb, spb)
        if e <= s:
            e = s + 1
        q.append((s, e, int(n.pitch)))

    by_s = defaultdict(list)
    for s, e, p in q:
        by_s[s].append((s, e, p))

    kept = [max(g, key=lambda x: ((x[1] - x[0]), x[2])) for g in by_s.values()]
    kept.sort(key=lambda x: (x[0], x[2]))

    mono = []
    for s, e, p in kept:
        if mono and s < mono[-1][1]:
            ps, pe, pp = mono[-1]
            mono[-1] = (ps, s, pp)
            if mono[-1][1] <= mono[-1][0]:
                mono.pop()
        mono.append((s, e, p))

    mono = [x for x in mono if x[1] > x[0]]
    return mono


def tokens_from_mono_with_pos(mono, max_rest=MAX_REST, max_dur=MAX_DUR, add_bar=True):
    def emit(out, prefix, v, cap):
        while v > cap:
            out.append(f"{prefix}_{cap}")
            v -= cap
        out.append(f"{prefix}_{v}")

    toks = []
    cur = 0
    for s, e, p in mono:
        if add_bar and (s % BAR_STEPS == 0):
            toks.append("BAR")
        while cur < s:
            if add_bar and (cur % BAR_STEPS == 0):
                toks.append("BAR")
            step = min(s - cur, max_rest)
            toks.append(f"REST_{step}")
            cur += step
        toks.append(f"POS_{s % STEPS_PER_BEAT}")
        toks.append(f"NOTE_{p}")
        dur = max(1, e - s)
        emit(toks, "DUR", dur, max_dur)
        cur = e

    return toks


def get_vocab():
    """Vocab and stoi/itos for encoding/decoding (matches notebook)."""
    vocab = [PAD, BOS, EOS, "BAR"]
    vocab += [f"POS_{i}" for i in range(STEPS_PER_BEAT)]
    vocab += [f"NOTE_{p}" for p in range(128)]
    vocab += [f"DUR_{d}" for d in range(1, MAX_DUR + 1)]
    vocab += [f"REST_{r}" for r in range(1, MAX_REST + 1)]
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    return vocab, stoi, itos


def midi_to_tokens(midi_path, spb=STEPS_PER_BEAT):
    """
    Load one MIDI file, take first instrument, clean to mono, return token list.
    """
    if miditoolkit is None:
        raise ImportError("miditoolkit is required. pip install miditoolkit")
    m = miditoolkit.MidiFile(midi_path)
    inst = m.instruments[0]
    mono = clean_to_monophonic(inst.notes, m.ticks_per_beat, spb=spb)
    return tokens_from_mono_with_pos(mono, add_bar=True)


def tokens_to_midi(
    tokens,
    out_path="generated.mid",
    tempo=140,
    steps_per_beat=STEPS_PER_BEAT,
):
    """Write a token sequence (with BAR, POS, NOTE, DUR, REST) to a MIDI file."""
    if miditoolkit is None:
        raise ImportError("miditoolkit is required. pip install miditoolkit")

    tpb = 480
    ticks_per_step = tpb // steps_per_beat

    midi = miditoolkit.MidiFile(ticks_per_beat=tpb)
    midi.tempo_changes = [miditoolkit.TempoChange(tempo, 0)]
    inst = miditoolkit.Instrument(program=56, is_drum=False, name="Trumpet")

    t = 0
    pending_pitch = None

    for tok in tokens:
        if tok in (PAD, BOS):
            continue
        if tok == EOS:
            break
        if tok == "BAR" or (tok.startswith("POS_") if isinstance(tok, str) else False):
            continue

        if "_" not in tok:
            continue
        typ, val = tok.split("_", 1)
        val = int(val)

        if typ == "REST":
            t += val
            pending_pitch = None
        elif typ == "NOTE":
            pending_pitch = val
        elif typ == "DUR" and pending_pitch is not None:
            start = t * ticks_per_step
            end = (t + val) * ticks_per_step
            inst.notes.append(
                miditoolkit.Note(velocity=90, pitch=pending_pitch, start=start, end=end)
            )
            t += val
            pending_pitch = None

    midi.instruments.append(inst)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    midi.dump(out_path)
    return out_path
