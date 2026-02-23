#!/usr/bin/env python3
"""
Generate a new jazz solo from a trained checkpoint.

Requires a checkpoint saved from the notebook (e.g. train in Colab, download best.pt to checkpoints/).
Optional: --chords-midi PATH uses chord_utils to restrict notes to chord/scale tones per bar.
"""
import argparse
import os
import sys

from paths import CHECKPOINTS_DIR, OUTPUTS_DIR
from midi_tokenizer import PAD, BOS, EOS, STEPS_PER_BEAT, tokens_to_midi
from chord_utils import get_chords_from_midi, allowed_midi_pitches_for_chord

import torch
import torch.nn as nn


class GPTMini(nn.Module):
    """Decoder-only transformer; must match notebook architecture."""
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, dropout=0.15, max_len=258):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, idx):
        B, T = idx.shape
        if T > self.max_len:
            raise ValueError(f"T={T} > max_len={self.max_len}")
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        causal = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        x = self.encoder(x, mask=causal)
        x = self.ln(x)
        return self.head(x)


def build_type_masks(vocab, stoi, device):
    def is_note(tok): return tok.startswith("NOTE_")
    def is_dur(tok):  return tok.startswith("DUR_")
    def is_rest(tok): return tok.startswith("REST_")
    def is_pos(tok):  return tok.startswith("POS_")
    banned = torch.zeros(len(vocab), dtype=torch.bool, device=device)
    banned[stoi[PAD]] = True
    banned[stoi[BOS]] = True
    masks = {}
    masks["NOTE"] = torch.tensor([is_note(t) for t in vocab], device=device) & ~banned
    masks["DUR"]  = torch.tensor([is_dur(t)  for t in vocab], device=device) & ~banned
    masks["REST"] = torch.tensor([is_rest(t) for t in vocab], device=device) & ~banned
    masks["POS"]  = torch.tensor([is_pos(t)  for t in vocab], device=device) & ~banned
    eos_mask = torch.zeros(len(vocab), dtype=torch.bool, device=device)
    eos_mask[stoi[EOS]] = True
    masks["EOS"] = eos_mask & ~banned
    bar_mask = torch.zeros(len(vocab), dtype=torch.bool, device=device)
    bar_mask[stoi["BAR"]] = True
    masks["BAR"] = bar_mask & ~banned
    return masks


def topk_sample(logits, k=20, temperature=0.9):
    logits = logits / max(temperature, 1e-6)
    if k is not None and k < logits.numel():
        v, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < v[-1], -1e9)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate_tokens(
    model,
    stoi,
    itos,
    masks,
    block_size,
    device,
    max_new_tokens=1200,
    temperature=0.9,
    top_k=20,
    chord_per_bar=None,
    pitch_min=54,
    pitch_max=84,
):
    """
    chord_per_bar: optional dict bar_index -> chord_figure. When set, NOTE sampling
    is restricted to chord/scale tones for the current bar.
    """
    allowed_by_prev = {
        "BOS":  (masks["BAR"] | masks["POS"]),
        "BAR":  (masks["REST"] | masks["POS"]),
        "REST": (masks["REST"] | masks["BAR"] | masks["POS"]),
        "POS":  masks["NOTE"],
        "NOTE": masks["DUR"],
        "DUR":  (masks["REST"] | masks["BAR"] | masks["POS"] | masks["EOS"]),
    }
    model.eval()
    ids = [stoi[BOS]]
    prev_type = "BOS"
    current_bar = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor(ids[-(block_size + 1):], device=device).unsqueeze(0)
            logits = model(x)[0, -1]
            allowed = allowed_by_prev.get(
                prev_type, (masks["REST"] | masks["POS"] | masks["BAR"] | masks["EOS"])
            )
            # Chord-guided note sampling: when about to emit a NOTE, restrict to chord tones
            if prev_type == "POS" and chord_per_bar is not None and current_bar in chord_per_bar:
                chord_fig = chord_per_bar[current_bar]
                allowed_pitches = allowed_midi_pitches_for_chord(
                    chord_fig, pitch_min=pitch_min, pitch_max=pitch_max, include_scale=True
                )
                chord_note_mask = torch.zeros_like(allowed, dtype=torch.bool, device=device)
                for p in allowed_pitches:
                    key = f"NOTE_{p}"
                    if key in stoi:
                        chord_note_mask[stoi[key]] = True
                allowed = allowed & chord_note_mask
                if not allowed.any():
                    allowed = allowed_by_prev["POS"]  # fallback if no overlap
            logits = logits.masked_fill(~allowed, -1e9)
            nxt = topk_sample(logits, k=top_k, temperature=temperature)
            ids.append(nxt)
            tok = itos[nxt]
            if tok == EOS:
                break
            prev_type = "BAR" if tok == "BAR" else tok.split("_")[0]
            if tok == "BAR":
                current_bar += 1
    return [itos[i] for i in ids]


def main():
    parser = argparse.ArgumentParser(description="Generate jazz solo MIDI from a trained checkpoint.")
    parser.add_argument("-c", "--checkpoint", default=os.path.join(CHECKPOINTS_DIR, "best.pt"),
                        help="Path to checkpoint .pt file")
    parser.add_argument("-o", "--out_dir", default=OUTPUTS_DIR, help="Directory to write generated .mid files")
    parser.add_argument("-n", "--num", type=int, default=1, help="Number of solos to generate")
    parser.add_argument("--max_tokens", type=int, default=1200, help="Max tokens per solo")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--tempo", type=int, default=140, help="MIDI tempo")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--chords-midi",
        default=None,
        metavar="PATH",
        help="MIDI file with chord symbols; generated notes are restricted to chord/scale tones per bar",
    )
    parser.add_argument("--pitch-min", type=int, default=54, help="Min MIDI pitch (default 54 = F#3 trumpet)")
    parser.add_argument("--pitch-max", type=int, default=84, help="Max MIDI pitch (default 84 = C6 trumpet)")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        print("Train in the notebook (JazzSoloGen_4.ipynb), then save best.pt to checkpoints/ and run again.", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    config = ckpt.get("config", {})
    block_size = config.get("block_size", 256)

    vocab_size = len(stoi)
    vocab = [itos[i] for i in range(vocab_size)]
    model = GPTMini(vocab_size, max_len=block_size + 2).to(device)
    model.load_state_dict(ckpt["model"])
    masks = build_type_masks(vocab, stoi, device)

    chord_per_bar = None
    if args.chords_midi:
        if not os.path.isfile(args.chords_midi):
            print(f"Chords MIDI not found: {args.chords_midi}", file=sys.stderr)
            sys.exit(1)
        chord_list = get_chords_from_midi(args.chords_midi)
        if not chord_list:
            print("Warning: no chord symbols found in", args.chords_midi, "- generating without chord conditioning", file=sys.stderr)
        else:
            chord_per_bar = {bar: fig for bar, fig in chord_list}
            print("Chord conditioning: {} chords over {} bars from {}".format(
                len(chord_per_bar), max(chord_per_bar.keys()) + 1, os.path.basename(args.chords_midi)))

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.num):
        tokens = generate_tokens(
            model, stoi, itos, masks, block_size, device,
            max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k,
            chord_per_bar=chord_per_bar,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max,
        )
        out_name = f"solo_{i + 1:03d}.mid"
        out_path = os.path.join(args.out_dir, out_name)
        tokens_to_midi(tokens, out_path=out_path, tempo=args.tempo, steps_per_beat=STEPS_PER_BEAT)
        print("Wrote", out_path)
    print("Done. Play with: python scripts/play_midi.py", args.out_dir)


if __name__ == "__main__":
    main()
