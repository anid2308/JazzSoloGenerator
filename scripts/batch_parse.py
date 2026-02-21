import os
import json
from music21 import converter, note, harmony

#optimizing proj paths for more flexible recalls
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
midi_dir = os.path.join(_PROJECT_ROOT, "data", "raw")
output_dir = os.path.join(_PROJECT_ROOT, "data", "processed")

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(midi_dir):
    if filename.endswith('.mid'):
        filepath = os.path.join(midi_dir, filename)
        print(f"processing: {filename}")

        try:
            score = converter.parse(filepath)
            events = []
            chords = []

            for n in score.flatten().notesAndRests:
                if isinstance(n, (note.Note, note.Rest)):
                    entry = {
                        "type": "note" if isinstance(n, note.Note) else "rest",
                        "pitch": str(n.pitch) if isinstance(n, note.Note) else None,
                        "midi": n.pitch.midi if isinstance(n, note.Note) else None,
                        "duration": float(n.quarterLength),
                        "offset": float(n.offset), 
                        "beat": float(n.beat),             
                        "measure": n.measureNumber 
                    }                
                    events.append(entry)

            for c in score.flat.getElementsByClass(harmony.ChordSymbol):
                chords.append({
                    "chord": c.figure,
                    "offset": float(c.offset),
                    "measure": c.measureNumber
                })

            output_data = {
                "events": events,
                "chords": chords
            }        

            output_filename = filename.replace('.mid', '.json')
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, 'w') as file:
                json.dump(output_data, file, indent=2)

        except Exception as e:
            print(f"Failed to parse {filename}: {e}")
