"""Parse a single MIDI file to JSON (pitch, midi, duration). Edit INPUT_NAME below and run."""
import json
import os
from music21 import converter

from paths import PROCESSED_DIR, RAW_DIR

INPUT_NAME = "autumn_leaves_chet_baker.mid"
score = converter.parse(os.path.join(RAW_DIR, INPUT_NAME))


sequence = []

for n in score.flatten().notes:
    if n.isNote:
        note_data = {
            "pitch": str(n.pitch),
            "midi": n.pitch.midi,
            "duration": float(n.quarterLength)
        }
        sequence.append(note_data)

os.makedirs(PROCESSED_DIR, exist_ok=True)
out_name = os.path.splitext(INPUT_NAME)[0] + "_solo.json"
with open(os.path.join(PROCESSED_DIR, out_name), "w") as f:
    json.dump(sequence, f)
print("Wrote", out_name)
