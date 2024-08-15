dict_chord_to_midi_pitches = {
    'N': [],
    'C:maj': [48, 52, 55], 'C:min': [48, 51, 55], 'C:dim': [48, 51, 54], 'C:aug': [48, 52, 56],
    'C:maj7': [48, 52, 55, 59], 'C:min7': [48, 51, 55, 58], 'C:7': [48, 52, 55, 58], 'C:sus2': [48, 50, 55], 'C:sus4': [48, 53, 55], 'C:maj6': [48, 52, 55, 57],
    'C#:maj': [49, 53, 56], 'C#:min': [49, 52, 56], 'C#:dim': [49, 52, 55], 'C#:aug': [49, 53, 57],
    'C#:maj7': [49, 53, 56, 48], 'C#:min7': [49, 52, 56, 59], 'C#:7': [49, 53, 56, 59], 'C#:sus2': [49, 51, 56], 'C#:sus4': [49, 54, 56], 'C#:maj6': [49, 53, 56, 58],
    'D:maj': [50, 54, 57], 'D:min': [50, 53, 57], 'D:dim': [50, 53, 56], 'D:aug': [50, 54, 58],
    'D:maj7': [50, 54, 57, 49], 'D:min7': [50, 53, 57, 48], 'D:7': [50, 54, 57, 48], 'D:sus2': [50, 52, 57], 'D:sus4': [50, 55, 57], 'D:maj6': [50, 54, 57, 59],
    'D#:maj': [51, 55, 58], 'D#:min': [51, 54, 58], 'D#:dim': [51, 54, 57], 'D#:aug': [51, 55, 59],
    'D#:maj7': [51, 55, 58, 50], 'D#:min7': [51, 54, 58, 49], 'D#:7': [51, 55, 58, 49], 'D#:sus2': [51, 53, 58], 'D#:sus4': [51, 56, 58], 'D#:maj6': [51, 55, 58, 48],
    'E:maj': [52, 56, 59], 'E:min': [52, 55, 59], 'E:dim': [52, 55, 58], 'E:aug': [52, 56, 48],
    'E:maj7': [52, 56, 59, 51], 'E:min7': [52, 55, 59, 50], 'E:7': [52, 56, 59, 50], 'E:sus2': [52, 54, 59], 'E:sus4': [52, 57, 59], 'E:maj6': [52, 56, 59, 49],
    'F:maj': [53, 57, 48], 'F:min': [53, 56, 48], 'F:dim': [53, 56, 59], 'F:aug': [53, 57, 49],
    'F:maj7': [53, 57, 48, 52], 'F:min7': [53, 56, 48, 51], 'F:7': [53, 57, 48, 51], 'F:sus2': [53, 55, 48], 'F:sus4': [53, 58, 48], 'F:maj6': [53, 57, 48, 50],
    'F#:maj': [54, 58, 49], 'F#:min': [54, 57, 49], 'F#:dim': [54, 57, 48], 'F#:aug': [54, 58, 50],
    'F#:maj7': [54, 58, 49, 53], 'F#:min7': [54, 57, 49, 52], 'F#:7': [54, 58, 49, 52], 'F#:sus2': [54, 56, 49], 'F#:sus4': [54, 59, 49], 'F#:maj6': [54, 58, 49, 51],
    'G:maj': [55, 59, 50], 'G:min': [55, 58, 50], 'G:dim': [55, 58, 49], 'G:aug': [55, 59, 51],
    'G:maj7': [55, 59, 50, 54], 'G:min7': [55, 58, 50, 53], 'G:7': [55, 59, 50, 53], 'G:sus2': [55, 57, 50], 'G:sus4': [55, 48, 50], 'G:maj6': [55, 59, 50, 52],
    'G#:maj': [56, 48, 51], 'G#:min': [56, 59, 51], 'G#:dim': [56, 59, 50], 'G#:aug': [56, 48, 52],
    'G#:maj7': [56, 48, 51, 55], 'G#:min7': [56, 59, 51, 54], 'G#:7': [56, 48, 51, 54], 'G#:sus2': [56, 58, 51], 'G#:sus4': [56, 49, 51], 'G#:maj6': [56, 48, 51, 53],
    'A:maj': [57, 49, 52], 'A:min': [57, 48, 52], 'A:dim': [57, 48, 51], 'A:aug': [57, 49, 53],
    'A:maj7': [57, 49, 52, 56], 'A:min7': [57, 48, 52, 55], 'A:7': [57, 49, 52, 55], 'A:sus2': [57, 59, 52], 'A:sus4': [57, 50, 52], 'A:maj6': [57, 49, 52, 54],
    'A#:maj': [58, 50, 53], 'A#:min': [58, 49, 53], 'A#:dim': [58, 49, 52], 'A#:aug': [58, 50, 54],
    'A#:maj7': [58, 50, 53, 57], 'A#:min7': [58, 49, 53, 56], 'A#:7': [58, 50, 53, 56], 'A#:sus2': [58, 48, 53], 'A#:sus4': [58, 51, 53], 'A#:maj6': [58, 50, 53, 55],
    'B:maj': [59, 51, 54], 'B:min': [59, 50, 54], 'B:dim': [59, 50, 53], 'B:aug': [59, 51, 55],
    'B:maj7': [59, 51, 54, 58], 'B:min7': [59, 50, 54, 57], 'B:7': [59, 51, 54, 57], 'B:sus2': [59, 49, 54], 'B:sus4': [59, 52, 54], 'B:maj6': [59, 51, 54, 56],
}

dict_chord_to_token = {
    'N': 0,
    'C:maj': 1, 'C:min': 2, 'C:dim': 3, 'C:aug': 4,
    'C:maj7': 5, 'C:min7': 6, 'C:7': 7, 'C:sus2': 8, 'C:sus4': 9, 'C:maj6': 10,
    'C#:maj': 11, 'C#:min': 12, 'C#:dim': 13, 'C#:aug': 14,
    'C#:maj7': 15, 'C#:min7': 16, 'C#:7': 17, 'C#:sus2': 18, 'C#:sus4': 19, 'C#:maj6': 20,
    'D:maj': 21, 'D:min': 22, 'D:dim': 23, 'D:aug': 24,
    'D:maj7': 25, 'D:min7': 26, 'D:7': 27, 'D:sus2': 28, 'D:sus4': 29, 'D:maj6': 30,
    'D#:maj': 31, 'D#:min': 32, 'D#:dim': 33, 'D#:aug': 34,
    'D#:maj7': 35, 'D#:min7': 36, 'D#:7': 37, 'D#:sus2': 38, 'D#:sus4': 39, 'D#:maj6': 40,
    'E:maj': 41, 'E:min': 42, 'E:dim': 43, 'E:aug': 44,
    'E:maj7': 45, 'E:min7': 46, 'E:7': 47, 'E:sus2': 48, 'E:sus4': 49, 'E:maj6': 50,
    'F:maj': 51, 'F:min': 52, 'F:dim': 53, 'F:aug': 54,
    'F:maj7': 55, 'F:min7': 56, 'F:7': 57, 'F:sus2': 58, 'F:sus4': 59, 'F:maj6': 60,
    'F#:maj': 61, 'F#:min': 62, 'F#:dim': 63, 'F#:aug': 64,
    'F#:maj7': 65, 'F#:min7': 66, 'F#:7': 67, 'F#:sus2': 68, 'F#:sus4': 69, 'F#:maj6': 70,
    'G:maj': 71, 'G:min': 72, 'G:dim': 73, 'G:aug': 74,
    'G:maj7': 75, 'G:min7': 76, 'G:7': 77, 'G:sus2': 78, 'G:sus4': 79, 'G:maj6': 80,
    'G#:maj': 81, 'G#:min': 82, 'G#:dim': 83, 'G#:aug': 84,
    'G#:maj7': 85, 'G#:min7': 86, 'G#:7': 87, 'G#:sus2': 88, 'G#:sus4': 89, 'G#:maj6': 90,
    'A:maj': 91, 'A:min': 92, 'A:dim': 93, 'A:aug': 94,
    'A:maj7': 95, 'A:min7': 96, 'A:7': 97, 'A:sus2': 98, 'A:sus4': 99, 'A:maj6': 100,
    'A#:maj': 101, 'A#:min': 102, 'A#:dim': 103, 'A#:aug': 104,
    'A#:maj7': 105, 'A#:min7': 106, 'A#:7': 107, 'A#:sus2': 108, 'A#:sus4': 109, 'A#:maj6': 110,
    'B:maj': 111, 'B:min': 112, 'B:dim': 113, 'B:aug': 114,
    'B:maj7': 115, 'B:min7': 116, 'B:7': 117, 'B:sus2': 118, 'B:sus4': 119, 'B:maj6': 120,
}

dict_key_to_token = {
    'C:maj': 121, 'C:min': 122,
    'C#:maj': 123, 'C#:min': 124,
    'D:maj': 125, 'D:min': 126,
    'D#:maj': 127, 'D#:min': 128,
    'E:maj': 129, 'E:min': 130,
    'F:maj': 131, 'F:min': 132,
    'F#:maj': 133, 'F#:min': 134,
    'G:maj': 135, 'G:min': 136,
    'G#:maj': 137, 'G#:min': 138,
    'A:maj': 139, 'A:min': 140,
    'A#:maj': 141, 'A#:min': 142,
    'B:maj': 143, 'B:min': 144,
}

dict_key_to_notes = {
    121: [1, 3, 5, 6, 8, 10, 12],   # C:maj  .
    122: [1, 3, 4, 6, 8, 9, 11],    # C:min  .
    123: [2, 4, 6, 7, 9, 11, 1],    # C#:maj .
    124: [2, 4, 5, 7, 9, 10, 12],   # C#:min .
    125: [3, 5, 7, 8, 10, 12, 2],   # D:maj  .
    126: [3, 5, 6, 8, 10, 11, 1],   # D:min  .
    127: [4, 6, 8, 9, 11, 1, 3],    # D#:maj .
    128: [4, 6, 7, 9, 11, 12, 2],   # D#:min . corrected
    129: [5, 7, 9, 10, 12, 2, 4],   # E:maj  .
    130: [5, 7, 8, 10, 12, 1, 3],   # E:min  .
    131: [6, 8, 10, 11, 1, 3, 5],   # F:maj  .
    132: [6, 8, 9, 11, 1, 2, 4],    # F:min  .
    133: [7, 9, 11, 12, 2, 4, 6],   # F#:maj .
    134: [7, 9, 10, 12, 2, 3, 5],   # F#:min .
    135: [8, 10, 12, 1, 3, 5, 7],   # G:maj  .
    136: [8, 10, 11, 1, 3, 4, 6],   # G:min  .
    137: [9, 11, 1, 2, 4, 6, 8],    # G#:maj .
    138: [9, 11, 12, 2, 4, 5, 7],   # G#:min . corrected
    139: [10, 12, 2, 3, 5, 7, 9],   # A:maj  .
    140: [10, 12, 1, 3, 5, 6, 8],   # A:min  .
    141: [11, 1, 3, 4, 6, 8, 10],   # A#:maj .
    142: [11, 1, 2, 4, 6, 7, 9],    # A#:min .
    143: [12, 2, 4, 5, 7, 9, 11],   # B:maj  .
    144: [12, 2, 3, 5, 7, 8, 10],   # B:min  .
}

def transpose_key(root, semitones):
    scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    root_index = scale.index(root)
    transposed_index = (root_index + semitones) % 12
    return scale[transposed_index]

def check_chords_against_key(dict_key_to_notes, dict_chord_to_midi_pitches):
    results = {}

    for chord, pitches in dict_chord_to_midi_pitches.items():
        if chord == 'N':
            continue

        # Process the chord to determine the correct key
        root, quality = chord.split(':')
        # Calculate the modulo 12 of the pitches
        mod_pitches = [p % 12 + 1 for p in pitches]
        if 'maj' in quality:
            key = root + ':maj'
        elif 'min' in quality:
            key = root + ':min'
        elif 'dim' in quality:
            transposed_root = transpose_key(root, -2)
            key = transposed_root + ':min'
        elif 'sus' in quality:
            key = root + ':maj'
        elif 'aug' in quality or '7' in quality:
            keys = [root + ':maj', root + ':min']
            # Get the correct scale notes from dict_key_to_notes
            if keys[0] in dict_key_to_token:
                scale_notes = set()
                scale_notes.update(dict_key_to_notes[dict_key_to_token[keys[0]]])
                scale_notes.update(dict_key_to_notes[dict_key_to_token[keys[1]]])

                # Check if all pitches are in the scale notes
                invalid_notes = [note for note in mod_pitches if note not in scale_notes]
                if invalid_notes:
                    results[chord] = invalid_notes
                else:
                    continue
        else:
            key = chord


        # Get the correct scale notes from dict_key_to_notes
        if key in dict_key_to_token:
            scale_notes = dict_key_to_notes[dict_key_to_token[key]]
            # Check if all pitches are in the scale notes
            invalid_notes = [note for note in mod_pitches if note not in scale_notes]
            if invalid_notes:
                results[chord] = invalid_notes

    return results

if __name__ == "__main__":
    errors = check_chords_against_key(dict_key_to_notes, dict_chord_to_midi_pitches)
    print("Augmented chords and 7 chords (such as C:7) are checked against both major and minor keys as they are not part of either of them.")
    print(errors) if errors else print("No errors found.")
