import numpy as np
import librosa
import pickle

# Load pre-trained sound classification model
model_path = "models/sound_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

def process_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    feature = np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0).reshape(1, -1)
    prediction = model.predict(feature)
    
    event_mapping = {
        0: "Fire alarm detected! Evacuate immediately.",
        1: "Possible intrusion detected! Alerting security.",
        2: "Baby crying detected. Notifying guardian.",
        3: "Doorbell detected. Please check the entrance.",
        4: "Gunshot detected! Take cover and call emergency services."
    }
    
    return event_mapping.get(prediction[0], "Unknown sound detected.")
