import os
import numpy as np
import csv
from scipy.io.wavfile import write

def generate_dummy_audio(filename, duration=3, fs=44100):
    t = np.linspace(0, duration, int(fs * duration))
    # Generate some noise + sine wave
    data = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))
    write(filename, fs, data.astype(np.float32))

os.makedirs('data', exist_ok=True)
os.makedirs('audio_samples', exist_ok=True)

labels = []
for i in range(10):  # 10 samples for testing
    label = np.random.choice([0, 1])  # 0=Truth, 1=Lie
    filename = f"data/sample_{i}.wav"
    generate_dummy_audio(filename)
    labels.append({'file': filename, 'label': label})

with open("data/labels.csv", mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['file', 'label'])
    writer.writeheader()
    writer.writerows(labels)

print("Dummy dataset generated in data/ folder")
