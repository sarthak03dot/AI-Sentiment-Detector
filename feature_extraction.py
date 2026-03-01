import sounddevice as sd
from scipy.io.wavfile import write
import whisper

def transcribe_audio(file_path="audio_samples/input.wav"):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""


def record_audio(filename="audio_samples/input.wav", duration=5, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    print("Recording saved:", filename)

import librosa
import numpy as np

def extract_features(file_path="audio_samples/input.wav", transcript=""):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # MFCC (Mean)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))

        # Pitch (Variance)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[pitches > 0]
        pitch_var = np.var(pitch_vals) if len(pitch_vals) > 0 else 0

        # Energy (Mean)
        energy_mean = np.mean(librosa.feature.rms(y=y))

        # Speech Rate (Words per minute)
        duration = librosa.get_duration(y=y, sr=sr)
        word_count = len(transcript.split()) if transcript else 0
        speech_rate = (word_count / duration) * 60 if duration > 0 else 0

        return [mfcc, pitch_var, energy_mean, speech_rate]
    except Exception as e:
        print(f"Error extracting features: {e}")
        return [0, 0, 0, 0]