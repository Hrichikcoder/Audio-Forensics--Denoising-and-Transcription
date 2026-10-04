import numpy as np
import wave
import noisereduce as nr
from scipy.signal import stft, istft, butter, filtfilt

def load_wav(path):
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr

def save_wav(path, audio, sr):
    audio = np.clip(audio, -1.0, 1.0)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((audio * 32767.0).astype(np.int16).tobytes())

print("Loading raw audio...")
audio, sr = load_wav("war-intercept.wav")

print("1. Performing STFT Spectral Deletion (No Ringing)...")
# Transform to image-like frequency domain
f, t, Zxx = stft(audio, fs=sr, nperseg=4096, noverlap=2048)

# Erase the exact jamming frequencies and their harmonics
jam_bands = [(690, 710), (1390, 1410), (2590, 2610), (2790, 2810), (4190, 4210)]
for low_f, high_f in jam_bands:
    idx = np.where((f >= low_f) & (f <= high_f))[0]
    Zxx[idx, :] *= 0.001  # Delete the frequencies

# Transform back to audio
_, audio_recovered = istft(Zxx, fs=sr)
audio = audio_recovered[:len(audio)] 

print("2. Applying AI Noise Reduction (Tape Hiss Removal)...")
audio = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.85, n_fft=2048)

print("3. Applying Voice Isolation Bandpass (300Hz - 3400Hz)...")
b, a = butter(4, [300 / (0.5 * sr), 3400 / (0.5 * sr)], btype='band')
audio = filtfilt(b, a, audio)

print("4. Applying Extreme Voice Boost (x15)...")
audio = np.tanh(audio * 15.0)

# Final safety normalization
max_val = np.max(np.abs(audio))
if max_val > 0:
    audio = (audio / max_val) * 0.95

save_wav("01warintercept.wav", audio, sr)
print("Done! Ultimate voice-boosted audio saved as 01warintercept.wav")