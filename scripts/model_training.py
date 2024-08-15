import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from chord_dataset import ChordDataset
from preprocess.dictionaries import dict_chord_to_midi_pitches, dict_chord_to_token, dict_key_to_token, dict_key_to_notes
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
from model import ChordGenLSTM
from torch.utils.data import DataLoader

# Number of chord tokens per token sequence (+1 for key token)
token_sequence_length = 8

# Reverse the dict_chord_to_token dictionary to map tokens back to chord names
token_to_chord_name = {v: k for k, v in dict_chord_to_token.items()}

def get_tokenized_file_paths(folder_path):
    """Retrieve all tokenized_chords.txt file paths in a folder and its subfolders."""
    tokenized_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file == 'tokenized_chords.txt':
                tokenized_files.append(os.path.join(root, file))
    return tokenized_files

def get_vocab_size():
    """Determine the vocabulary size from chord fragments."""
    return max(dict_key_to_token.values()) + 1

def prepare_dataset():
    """Prepare the dataset for training."""
    all_tokenized_paths = get_tokenized_file_paths('..\\data\\POP909')
    chord_fragments = []

    for file_path in all_tokenized_paths:
        with open(file_path, 'r') as file:
            try:
                tokens = list(map(int, file.read().strip().split()))
            except ValueError as e:
                print(f"Error processing file: {file_path}")
                print(f"Line content: {file.read().strip().split()}")
                raise e
            # Filter out zeros
            tokens = [token for token in tokens if token != 0]

            # Ensure there are at least token_sequence_length tokens plus one key token
            if len(tokens) < token_sequence_length + 1:
                continue

            key_token = tokens[0]

            # Create groups of 9 tokens composed of 8 chord tokens with 1 key token at the beginning
            for i in range(1, len(tokens) - token_sequence_length + 1, token_sequence_length):
                chord_fragment = [key_token] + tokens[i:i + token_sequence_length]
                chord_fragments.append(chord_fragment)

    dataset = ChordDataset(chord_fragments)

    with open('dataset_8chords_tokens.pkl', 'wb') as pickle_file:
        pickle.dump(dataset, pickle_file)
    
    return dataset

def key_aware_loss(outputs, targets, criterion, penalty=1.0):
    # Standard cross-entropy loss
    ce_loss = criterion(outputs, targets)

    # Initialize penalty loss
    penalty_loss = 0.0
    
    # Iterate over each sequence in the batch
    for i in range(0, len(targets), 9):  # Step by 9 to process each sequence
        # First token is the key, take it
        main_key = targets[i].item()
        
        # Adjust the main key to minor if it's major, or to major if it's minor
        complement_key = main_key + 1 if main_key % 2 == 1 else main_key - 1

        # Iterate over the chords in this sequence
        for j in range(i + 1, i + 9):
            chord_token = targets[j].item()

            # Check if the chord token has a corresponding chord name
            if chord_token in token_to_chord_name:
                chord_name = token_to_chord_name[chord_token]

                # Get the MIDI pitches for the chord
                midi_notes = dict_chord_to_midi_pitches[chord_name]
                
                keys_to_check = [main_key]

                # If it's an augmented or dominant 7th chord, also check against the parallel major/minor key
                if chord_name.endswith('aug') or (chord_name.endswith('7') and not chord_name.endswith('maj7')):
                    keys_to_check.append(complement_key)
                
                # Count how many notes are off-key
                off_notes = 0
                for note in midi_notes:
                    # Get the scale degree of the note (0 to 11)
                    scale_degree = note % 12 + 1
                    
                    # Check against all relevant keys
                    if not any(scale_degree in dict_key_to_notes[k] for k in keys_to_check):
                        off_notes += 1
                
                # Add a fraction of the penalty proportional to the number of off notes
                if off_notes > 0:
                    penalty_loss += (off_notes / len(midi_notes)) * penalty
    
    # Normalize penalty loss by the number of sequences in the batch
    penalty_loss = penalty_loss / (len(targets) / 9)
    total_loss = ce_loss + penalty_loss
    return total_loss

def train_model(model, device, dataset, use_key_aware, epochs=25, batch_size=4, learning_rate=0.001):
    """Train the model with the provided dataset."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # No penalty applied in the first 5 epochs
        penalty = 2 if epoch >= 5 else 0.0

        for chords in dataloader:
            chords = chords.to(device)
            optimizer.zero_grad()
            outputs = model(chords)
            outputs = outputs.view(-1, outputs.size(2))

            # Create target with the key followed by circular left-shifted chords (by 1)
            # {key c1 c2 c3 c4 c5 c6 c7 c8}
            key_token = chords[:, 0].unsqueeze(1)
            chords_out = chords[:, 1:]
            chords_out_shifted = torch.cat((chords_out[:, 1:], chords_out[:, :1]), dim=1)
            chords_out_new = torch.cat((key_token, chords_out_shifted), dim=1).view(-1).to(device)
            
            if use_key_aware:
                # Calculate loss with key-aware penalty
                loss = key_aware_loss(outputs, chords_out_new, criterion, penalty=penalty)
            else:
                # Calculate normal cross-entropy loss
                loss = criterion(outputs, chords_out_new)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Save the model at different training stages
        if (epoch + 1) == 5 or (epoch + 1) == 15:
            checkpoint_path = f'./trained_models/model_epoch_{epoch + 1}.pth' if use_key_aware else f'./trained_models/model_no_key_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Model checkpoint saved to {checkpoint_path}')
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}')

def train_and_save_model(dataset, vocab_size, use_key_aware=True):
    """Create, train, and save the model."""
    model = ChordGenLSTM(vocab_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    train_model(model, device, dataset, use_key_aware)

    model_save_path = './trained_models/model.pth' if use_key_aware else './trained_models/model_no_key.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

def sample_with_temperature(logits, temperature=1.0):
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Apply softmax to get probabilities
    probabilities = torch.softmax(scaled_logits, dim=-1)

    # Initialize a list to store sampled tokens
    sampled_tokens = []

    # Loop over each time step (sequence length)
    for i in range(probabilities.size(1)):  # probabilities.size(1) is the sequence length
        prob_dist = probabilities[:, i, :]  # Get the probability distribution for each step
        sampled_token = torch.multinomial(prob_dist, num_samples=1)
        sampled_tokens.append(sampled_token)

    # Concatenate sampled tokens along the sequence dimension
    sampled_tokens = torch.cat(sampled_tokens, dim=1)
    return sampled_tokens

def load_and_generate_chords(vocab_size, chord_data, temperature, model_path='./trained_models/model.pth'):
    """Load the model and generate chords from chord data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ChordGenLSTM(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded and set to evaluation mode")
    
    processed_chord_data = torch.tensor([chord_data]).to(device)
    print(f"Given chord data tokens: {processed_chord_data.tolist()}")

    with torch.no_grad():
        generated_chord_logits = model(processed_chord_data)
    generated_chords = sample_with_temperature(generated_chord_logits, temperature=temperature).squeeze().cpu().numpy()

    print(f"Generated chord tokens: {[generated_chords.tolist()]}")

    return generated_chords

def tokens_to_midi(chord_tokens, output_file='output.mid'):
    # Translate token to chord string
    token_to_chord = {v: k for k, v in dict_chord_to_token.items()}
    token_to_key = {v: k for k, v in dict_key_to_token.items()}
    
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Set tempo (120 BPM)
    tempo = bpm2tempo(120)
    track.append(MetaMessage('set_tempo', tempo=tempo))
    
    # Time signature (4/4)
    track.append(MetaMessage('time_signature', numerator=4, denominator=4))
    
    # Duration of one chord in ticks
    ticks_per_beat = midi.ticks_per_beat
    chord_duration_ticks = 4 * ticks_per_beat
    
    for token_seq in chord_tokens:
        # Extract the key token from the first token of the sequence
        key_token = token_seq[0]
        
        if key_token in token_to_key:
            key_str = token_to_key[key_token]
            print(f"Key: {key_str}")
            # Add key signature meta message
            track.append(MetaMessage('text', text=key_str, time=0))
        
        # Process the remaining eight tokens as chord tokens
        for token in token_seq[1:]:
            if token not in token_to_chord:
                continue
            chord_str = token_to_chord[token]
            midi_notes = dict_chord_to_midi_pitches[chord_str]
            
            # Add note_on events for the chord
            for note in midi_notes:
                track.append(Message('note_on', note=note, velocity=96, time=0))
                
            first_note = True
            for note in midi_notes:
                if first_note:
                    track.append(Message('note_off', note=note, velocity=96, time=chord_duration_ticks))
                    first_note = False
                else:
                    track.append(Message('note_off', note=note, velocity=96, time=0))
    
    midi.save(output_file)
    print(f'MIDI file saved as {output_file}')

if __name__ == "__main__":
    vocab_size = get_vocab_size()
    chord_data = [133, 111, 11, 102, 32]
    temperature = 1.0

    if not os.path.exists('./dataset_8chords_tokens.pkl'):
        dataset = prepare_dataset()
        print(f"\n***Dataset created and vocabulary size determined ({vocab_size}).***\n")

        train_and_save_model(dataset, vocab_size)
        print("\n***Model prepped and trained.***\n")

    elif not os.path.exists('./trained_models/model.pth'):
        with open('dataset_8chords_tokens.pkl', 'rb') as pickle_file:
            dataset = pickle.load(pickle_file)
        print(f"\n***Dataset loaded and vocabulary size determined ({vocab_size}).***\n")

        train_and_save_model(dataset, vocab_size)
        print("\n***Model prepped and trained.***\n")

    elif not os.path.exists('./trained_models/model_no_key.pth'):
        with open('dataset_8chords_tokens.pkl', 'rb') as pickle_file:
            dataset = pickle.load(pickle_file)
        print(f"\n***Dataset loaded and vocabulary size determined ({vocab_size}).***\n")

        train_and_save_model(dataset, vocab_size, use_key_aware=False)
        print("\n***Model prepped and trained.***\n")

    else:
        print(f"\n***Dataset already created (vocab size: {vocab_size}) and model already trained.***\n")

    midi_tokens = load_and_generate_chords(vocab_size, chord_data, temperature)
    tokens_to_midi([midi_tokens])