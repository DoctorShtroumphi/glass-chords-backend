import numpy as np
import random
import torch

from collections import defaultdict
from preprocess.dictionaries import dict_chord_to_midi_pitches, dict_chord_to_token, dict_key_to_token, dict_key_to_notes
from model import ChordGenLSTM

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

def load_and_generate_chords(vocab_size, chord_data_options, temperature, model_path='./trained_models/model.pth', num_sequences=100):
    """Load the model and generate multiple sequences of chords from chord data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChordGenLSTM(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    generated_sequences = []

    for _ in range(num_sequences):
        chord_data = random.choice(chord_data_options)
        processed_chord_data = torch.tensor([chord_data]).to(device)
        with torch.no_grad():
            generated_chord_logits = model(processed_chord_data)
        generated_chords = sample_with_temperature(generated_chord_logits, temperature=temperature).squeeze().cpu().numpy()
        generated_sequences.append(generated_chords)

    return generated_sequences

def transpose_key(root, semitones):
    scale = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    root_index = scale.index(root)
    transposed_index = (root_index + semitones) % 12
    return scale[transposed_index]

def check_chords_against_key(generated_chords):
    """Check generated chords against the key and return off-key results."""
    results = {}
    
    for chord_token in generated_chords[1:]:
        chord_name = token_to_chord_name.get(chord_token)
        if not chord_name or chord_name == 'N':
            continue

        _, quality = chord_name.split(':')
        if 'aug' in quality or '7' in quality:
            continue

        mod_pitches = [p % 12 + 1 for p in dict_chord_to_midi_pitches[chord_name]]

        scale_notes = dict_key_to_notes[generated_chords[0]]
        invalid_notes = [note for note in mod_pitches if note not in scale_notes]
        
        if invalid_notes:
            results[chord_name] = invalid_notes

    return results

def evaluate_models(vocab_size, chord_data_options, temperature, runs):
    """Evaluate both models by generating chords and calculating the percentage of off-key chords."""
    model_paths = {
        'key_aware_model': './trained_models/model.pth',
        'no_key_aware_model': './trained_models/model_no_key.pth'
    }
    
    results = defaultdict(list)
    
    for model_name, model_path in model_paths.items():
        generated_sequences = load_and_generate_chords(vocab_size, chord_data_options, temperature, model_path, num_sequences=runs)
        
        for generated_chords in generated_sequences:
            off_key_results = check_chords_against_key(generated_chords)
            off_key_count = len(off_key_results)
            results[model_name].append(off_key_count)
    
    percentages = {model_name: (np.mean(result) / 4) * 100 for model_name, result in results.items()}
    
    return percentages

vocab_size = max(dict_key_to_token.values()) + 1
chord_data_options = [
    [130, 1, 42, 21, 21], 
    [133, 111, 11, 102, 32], 
    [138, 82, 12, 61, 32]
]
temperature = 1.0

token_to_chord_name = {v: k for k, v in dict_chord_to_token.items()}

print(check_chords_against_key(chord_data_options[0]))
print(check_chords_against_key(chord_data_options[1]))
print(check_chords_against_key(chord_data_options[2]))

percentages = evaluate_models(vocab_size, chord_data_options, temperature, 10000)
print(f"Percentage of off-key chords - Key-aware model: {percentages['key_aware_model']}%")
print(f"Percentage of off-key chords - No-key-aware model: {percentages['no_key_aware_model']}%")
