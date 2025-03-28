import librosa
import numpy as np
import pickle

# Load the trained sound classification model
with open("models/sound_model.pkl", "rb") as f:
    model = pickle.load(f)

def classify_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    features = np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0).reshape(1, -1)
    prediction = model.predict(features)
    
    return prediction
