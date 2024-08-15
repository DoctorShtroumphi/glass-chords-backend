import logging
import torch

from preprocess.dictionaries import dict_chord_to_midi_pitches, dict_chord_to_token, dict_key_to_token, dict_key_to_notes
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import ChordGenLSTM

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load models
vocab_size = max(dict_key_to_token.values()) + 1

normal_model = ChordGenLSTM(vocab_size).to('cpu')
normal_model.load_state_dict(torch.load('./model-code/scripts/trained_models/model.pth', map_location=torch.device('cpu')))

lesser_trained_model_5 = ChordGenLSTM(vocab_size).to('cpu')
lesser_trained_model_5.load_state_dict(torch.load('./model-code/scripts/trained_models/model_epoch_5.pth', map_location=torch.device('cpu')))

lesser_trained_model_15 = ChordGenLSTM(vocab_size).to('cpu')
lesser_trained_model_15.load_state_dict(torch.load('./model-code/scripts/trained_models/model_epoch_15.pth', map_location=torch.device('cpu')))

model_no_key_aware = ChordGenLSTM(vocab_size).to('cpu')
model_no_key_aware.load_state_dict(torch.load('./model-code/scripts/trained_models/model_no_key.pth', map_location=torch.device('cpu')))

# Set models to evaluation mode
normal_model.eval()
lesser_trained_model_5.eval()
lesser_trained_model_15.eval()
model_no_key_aware.eval()

def generate_chords(model, key, chords, temperature=1.0):
    """Generate chords using the specified model and parameters."""
    # Prepare the input data
    chord_data = torch.tensor([key] + chords).unsqueeze(0)
    with torch.no_grad():
        outputs = model(chord_data)
        logits = outputs.view(-1, outputs.size(-1))
        if temperature != 1.0:
            logits /= temperature
        probabilities = torch.softmax(logits, dim=-1)
        generated_chords = torch.multinomial(probabilities, num_samples=1).squeeze().tolist()
    return generated_chords

@app.route('/generate_chords', methods=['POST'])
def generate_chords_route():
    # Extract parameters from the request
    data = request.json
    key = data.get('key')
    chords = data.get('chords')
    model_type = data.get('model_type')
    temperature = data.get('temperature', 1.0)
    key_aware_loss_enabled = data.get('key_aware_loss_enabled')

    # Log the input parameters
    logging.info(f"Model Type: {model_type}, Temperature: {temperature}, Key Aware Enabled: {key_aware_loss_enabled}")
    logging.info(f"Input - Key: {key}, Chords: {chords}")

    # Input validation
    if not isinstance(key, int) or not all(isinstance(chord, int) for chord in chords) or len(chords) != 4:
        return jsonify({"error": "Invalid input. 'key' must be an integer and 'chords' must be a list of 4 integers."}), 400

    # Select the model based on the scenario
    if key_aware_loss_enabled:
        if model_type == 'normal':
            model = normal_model
        elif model_type == 'least_trained':
            model = lesser_trained_model_5
        elif model_type == 'mid_trained':
            model = lesser_trained_model_15
        else:
            return jsonify({"error": f"Invalid 'model_type': {model_type}"}), 400
    else:
        if model_type == 'normal':
            model = model_no_key_aware
        elif model_type == 'least_trained':
            model = model_no_key_aware
        elif model_type == 'mid_trained':
            model = model_no_key_aware
        else:
            return jsonify({"error": f"Invalid 'model_type': {model_type}"}), 400
    
    # Generate chords
    generated_chords = generate_chords(model, key, chords, temperature)
    
    # Log the generated output
    logging.info(f"Output - Key: {generated_chords[0]}, Generated Chords: {generated_chords[1:]}")

    # Return the generated chords
    return jsonify({"generated_chords": generated_chords})

@app.route('/chord_data', methods=['GET'])
def get_chord_data():
    return jsonify({
        "chord_to_midi_pitches": dict_chord_to_midi_pitches,
        "chord_to_token": dict_chord_to_token,
        "key_to_token": dict_key_to_token,
        "key_to_notes": dict_key_to_notes
    })

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
