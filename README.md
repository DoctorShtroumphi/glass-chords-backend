# GlassChords - Backend API

## Project Overview

This is the backend API for the GlassChords project. It serves as the core engine for generating AI-powered chord progressions based on user inputs. Built with Python and Flask, the API processes requests from the frontend, applies the selected model and settings, and returns the generated chords.

## Features

- **Chord Generation**: Generates chord sequences based on input chords and user-selected parameters.
- **Model Selection**: Choose from multiple models with varying levels of training and key awareness.
- **Temperature Control**: Adjust the creativity and randomness of the generated chords.
- **Key Awareness**: Option to enforce key-awareness in the generated progressions.

## Model

The AI model used in this project is a simple, lightweight Long Short-Term Memory (LSTM) neural network. The model is designed to generate chord progressions based on the input key and chords. It has been trained on a dataset of chord sequences, and different versions of the model are provided based on varying levels of training. Additionally, the model can be used with or without a custom "key aware loss function" that influences how closely the generated chords adhere to the selected musical key.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/glasschords-backend.git
   cd glasschords-backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Flask server:
   ```bash
   flask run
   ```

## Usage

1. The API serves at `http://localhost:5000`.

2. **Endpoints:**
   - **`/generate_chords`**: Accepts a POST request with the following JSON payload:
     ```json
     {
       "key": 133,
       "chords": [139, 141, 144, 142],
       "model_type": "normal",
       "temperature": 1.0,
       "key_aware_loss_enabled": true
     }
     ```
     Returns the generated chord sequence based on the provided parameters.

   - **`/chord_data`**: Accepts a GET request and returns all the necessary dictionaries needed by the frontend, including chord-to-token mappings, key-to-notes mappings, and other relevant data.

3. The API responds with the generated chord sequence or the required data for the frontend.

## Technologies Used

- **Python**: Core language.
- **Flask**: Web framework for building the API.
- **PyTorch**: Used for building and running the AI model (LSTM).
- **Gunicorn**: WSGI server for deploying the API.

## Deployment

The backend API is deployed using [Heroku](https://www.heroku.com/). The deployment process involves pushing the code to Heroku's Git repository, where the application is automatically built and run using Gunicorn as the WSGI server with the following command:
```bash
gunicorn generate_api:app
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.