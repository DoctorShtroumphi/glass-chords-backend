import os

from dictionaries import dict_chord_to_midi_pitches, dict_chord_to_token, dict_key_to_token
from mido import MidiFile, MidiTrack, Message, second2tick

def get_all_midi_paths():
    root_dir = '..\\..\\data\\POP909'
    midi_paths = []
    for song_idx in range(1, 910):
        song_dir = os.path.join(root_dir, f'{song_idx:03d}')
        midi_file_path = os.path.join(song_dir, f'{song_idx:03d}.mid')
        chord_file_path = os.path.join(song_dir, 'chord_midi.txt')
        key_file_path = os.path.join(song_dir, 'key_audio.txt')
        if os.path.exists(midi_file_path) and os.path.exists(chord_file_path):
            midi_paths.append((midi_file_path, chord_file_path, key_file_path))
    return midi_paths

# Function to convert a chord to MIDI pitch notes
def chord_to_notes(chord):
    # If the chord is not in the dictionary, try to find the closest match
    if chord not in dict_chord_to_midi_pitches:
        # Remove inversions and extensions (e.g., /5, /7, etc.)
        base_chord = chord.split('/')[0]
        # Find the closest match in the dictionary
        closest_chord = None
        for key in dict_chord_to_midi_pitches.keys():
            if key.startswith(base_chord):
                closest_chord = key
                break
        if closest_chord:
            return dict_chord_to_midi_pitches[closest_chord]
        else:
            # Return an empty list if no match is found
            return []
    else:
        return dict_chord_to_midi_pitches[chord]
    
# Function to convert a chord to a token
def chord_to_token(chord):
    # Map flat chords to their sharp equivalents
    flat_to_sharp = {
        'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
    }

    for flat, sharp in flat_to_sharp.items():
        chord = chord.replace(flat, sharp)

    # Convert hdim* to dim (where * is any number)
    if 'hdim' in chord:
        chord = chord.split('hdim')[0] + 'dim'

    # If the chord is still not in the dictionary, try to find the closest match
    if chord not in dict_chord_to_token:
        
        # Remove inversions and extensions (e.g., /5, /7, etc.)
        base_chord = chord.split('/')[0]
        # Find the closest match in the dictionary
        closest_chord = None
        for key in dict_chord_to_token.keys():
            if key.startswith(base_chord):
                closest_chord = key
                break
        if closest_chord:
            return dict_chord_to_token[closest_chord]
        else:
            # Return 0 if no match is found (assuming 0 for unrecognized chords)
            return 0
    else:
        return dict_chord_to_token[chord]

# Function to convert a key to a token
def key_to_token(key):
    # Map flat keys to their sharp equivalents
    flat_to_sharp = {
        'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'
    }

    for flat, sharp in flat_to_sharp.items():
        key = key.replace(flat, sharp)
    
    return dict_key_to_token[key]

# Function to get ticks per beat and tempo from a midi file
def get_time_info(midi_file_path):
    mid = MidiFile(midi_file_path)
    ticks_per_beat = mid.ticks_per_beat
    time_sig = None

    tempo_changes = []
    current_time = 0

    for msg in mid.tracks[0]:
        if msg.type == 'set_tempo':
            tempo_changes.append((current_time, msg.tempo))
            current_time += msg.time
        if msg.type == 'time_signature':
            time_sig = msg

    return ticks_per_beat, tempo_changes, time_sig

# Function to calculate the current tempo at a specific tick (handles tempo changes)
def get_tempo_at_tick(tick, tempo_changes):
    # Default MIDI tempo
    current_tempo = 500000
    for time, tempo in tempo_changes:
        if time > tick:
            break
        current_tempo = tempo
    return current_tempo

# Function to convert text file to MIDI
def text_to_midi(file_path, ticks_per_beat, tempo_changes, time_sig):
    midi = MidiFile()
    midi.ticks_per_beat = ticks_per_beat

    track = MidiTrack()
    track.append(time_sig)
    midi.tracks.append(track)

    with open(file_path, 'r') as f:
        accumulated_ticks_skipped = 0

        for line in f:
            start, end, chord = line.strip().split()
            # Calculate the ticks for start and end times considering tempo changes
            start_ticks = second2tick(float(start), ticks_per_beat, tempo_changes[0][1])
            end_ticks = second2tick(float(end), ticks_per_beat, tempo_changes[0][1])
            duration = end_ticks - start_ticks
            
            if chord == 'N':
                # Accumulate empty time
                accumulated_ticks_skipped += duration
                continue

            notes = chord_to_notes(chord)

            first_note = True
            for note in notes:
                if first_note:
                    track.append(Message('note_on', note=note, velocity=96, time=accumulated_ticks_skipped))
                    accumulated_ticks_skipped = 0
                else:
                    track.append(Message('note_on', note=note, velocity=96, time=0))
            
            first_note = True
            for note in notes:
                if first_note:
                    track.append(Message('note_off', note=note, velocity=96, time=duration))
                    first_note = False
                else:
                    track.append(Message('note_off', note=note, velocity=96, time=0))
    
    return midi

# Function to convert text file to tokenized chord file
def text_to_tokenized_chords(text_chord_file_path, key_file_path, output_path):
    tokens = []
    with open(key_file_path, 'r') as f:
        lines = f.readlines()
    
    _, _, key = lines[0].strip().split()
    token = str(key_to_token(key))
    tokens.append(token)

    with open(text_chord_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        _, _, chord = line.strip().split()
        token = str(chord_to_token(chord))
        tokens.append(token)

    with open(output_path, 'w') as out_f:
        out_f.write(' '.join(tokens))

file_paths = get_all_midi_paths()

for (midi_file_path, chord_file_path, key_file_path) in file_paths:
    # CHORD TEXT DATA TO MIDI (MIDI TECHNIQUE)
    # ticks_per_beat, tempo_changes, time_sig = get_time_info(midi_file_path)
    # midi = text_to_midi(chord_file_path, ticks_per_beat, tempo_changes, time_sig)
    # midi.save(chord_file_path.replace('.txt', '.mid'))

    # CHORD TEXT DATA TO TOKENIZED TEXT
    text_to_tokenized_chords(chord_file_path, key_file_path, os.path.join(os.path.dirname(chord_file_path), 'tokenized_chords.txt'))