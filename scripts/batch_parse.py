import json
import os
from music21 import converter, note, harmony

from paths import PROCESSED_DIR, RAW_DIR

os.makedirs(PROCESSED_DIR, exist_ok=True)

for filename in sorted(os.listdir(RAW_DIR)):
    if filename.lower().endswith((".mid", ".midi")):
        filepath = os.path.join(RAW_DIR, filename)
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

            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(PROCESSED_DIR, output_filename)

            with open(output_path, 'w') as file:
                json.dump(output_data, file, indent=2)

        except Exception as e:
            print(f"Failed to parse {filename}: {e}")
