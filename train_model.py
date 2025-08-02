import os
import pickle
import numpy as np
from extract_features import extract_features_from_audio
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "voice_data"
X = []
y = []

for emotion_label in os.listdir(DATA_DIR):
    emotion_dir = os.path.join(DATA_DIR, emotion_label)
    for file in os.listdir(emotion_dir):
        if file.endswith(".wav"):
            try:
                file_path = os.path.join(emotion_dir, file)
                features = extract_features_from_audio(file_path)
                X.append(features)
                y.append(emotion_label)
            except Exception as e:
                print(f"[WARNING] Skipped {file}: {e}")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("[INFO] Model trained and saved as emotion_model.pkl")
