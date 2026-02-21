import json
import os
from music21 import converter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")

score = converter.parse(os.path.join(RAW_DIR, "autumn_leaves_chet_baker.mid"))


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
with open(os.path.join(PROCESSED_DIR, "AL_chet_baker_solo.json"), "w") as file:
    json.dump(sequence, file)
