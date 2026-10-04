import numpy as np
import wave
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import spectrogram

def load_wav(path):
    with wave.open(path, 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32), w.getframerate()

print("Loading audio files...")
raw_audio, sr = load_wav("war-intercept.wav")
clean_audio, _ = load_wav("01warintercept.wav")

# ---------------------------------------------------------
# GRAPH 1: Standard 2D FFT Spectrum
# ---------------------------------------------------------
print("Generating 1. Standard 2D FFT Spectrum...")
freqs = np.fft.rfftfreq(len(raw_audio), d=1/sr)
raw_mag = 20 * np.log10(np.abs(np.fft.rfft(raw_audio)) + 1e-9)
clean_mag = 20 * np.log10(np.abs(np.fft.rfft(clean_audio)) + 1e-9)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(freqs, raw_mag, color='crimson', linewidth=0.5)
axes[0].set_title('RAW AUDIO: Jamming Peaks at 700Hz and 2600Hz')
axes[0].set_xlim(0, 5000); axes[0].grid(True, alpha=0.3)
axes[0].set_ylabel('Magnitude (dB)')

axes[1].plot(freqs, clean_mag, color='steelblue', linewidth=0.5)
axes[1].set_title('CLEANED AUDIO: Jamming Peaks Erased via STFT')
axes[1].set_xlim(0, 5000); axes[1].grid(True, alpha=0.3)
axes[1].set_ylabel('Magnitude (dB)'); axes[1].set_xlabel('Frequency (Hz)')

plt.tight_layout()
plt.savefig("02fig1_fft.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# GRAPH 2: Standard 2D Spectrogram (Time vs Frequency)
# ---------------------------------------------------------
print("Generating 2. Standard 2D Spectrograms...")
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
for ax, audio, title, cmap in zip(axes, [raw_audio, clean_audio], ['RAW AUDIO', 'CLEANED AUDIO'], ['Reds_r', 'Blues_r']):
    segment = audio[:sr*30] # First 30 seconds
    f, t, Sxx = spectrogram(segment, fs=sr, nperseg=1024, noverlap=512)
    mask = f <= 5000
    ax.pcolormesh(t, f[mask], 10*np.log10(Sxx[mask] + 1e-12), shading='gouraud', cmap=cmap, vmin=-80, vmax=0)
    ax.set_title(title); ax.set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig("02fig2_spectrogram.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# GRAPH 3: The 3D FFT Waterfall Plot
# ---------------------------------------------------------
print("Generating 3. 3D FFT Transform Graphs (This may take a moment)...")
fig = plt.figure(figsize=(14, 6))

# We take a small 1.5 second snippet so the 3D render doesn't crash the computer
snippet_raw = raw_audio[sr*10 : int(sr*11.5)]
snippet_clean = clean_audio[sr*10 : int(sr*11.5)]

for i, (audio, title, color_map) in enumerate(zip([snippet_raw, snippet_clean], 
                                                  ['3D FFT: Raw Audio (Notice the Jamming Walls)', '3D FFT: Cleaned Audio (Voice Only)'], 
                                                  ['magma', 'viridis'])):
    
    f, t, Sxx = spectrogram(audio, fs=sr, nperseg=512, noverlap=256)
    
    # Filter to only show 0 to 4000 Hz for better visibility
    mask = (f >= 0) & (f <= 4000)
    f_masked = f[mask]
    Sxx_masked = 10 * np.log10(Sxx[mask, :] + 1e-10)
    
    # Create 3D Meshgrid
    T, F = np.meshgrid(t, f_masked)
    
    ax = fig.add_subplot(1, 2, i+1, projection='3d')
    
    # Plot the 3D surface
    surf = ax.plot_surface(T, F, Sxx_masked, cmap=color_map, edgecolor='none', alpha=0.9)
    
    ax.set_title(title, pad=20)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Freq (Hz)')
    ax.set_zlabel('Magnitude (dB)')
    
    # Adjust viewing angle for best 3D perspective
    ax.view_init(elev=35, azim=-45)

plt.tight_layout()
plt.savefig("02fig3_3D_FFT.png", dpi=200)
plt.close()

print("Done! Outputs saved as 02fig1_fft.png, 02fig2_spectrogram.png, and 02fig3_3D_FFT.png")