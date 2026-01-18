import json
from music21 import converter


score = converter.parse('data/raw_midis/autumn_leaves_chet_baker_solo.mid')


sequence = []

for n in score.flatten().notes:
    if n.isNote:
        note_data = {
            "pitch": str(n.pitch),
            "midi": n.pitch.midi,
            "duration": float(n.quarterLength)
        }
        sequence.append(note_data)

with open('data/processed/AL_chet_baker_solo.json', 'w') as file:
    json.dump(sequence, file)