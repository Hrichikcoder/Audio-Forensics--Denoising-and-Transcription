# Audio Forensics: Denoising and Transcription

This project provides a comprehensive audio forensics pipeline designed to clean, analyze, and transcribe highly corrupted or jammed audio signals. It uses advanced digital signal processing techniques, artificial intelligence for noise reduction, and OpenAI's Whisper model for speech-to-text transcription.

## Features

*   **Advanced Audio Cleanup (`01audiocleanup.py`)**: 
    *   **STFT Spectral Deletion**: Precisely targets and deletes jamming frequencies (e.g., 700Hz, 1400Hz, 2600Hz) and their harmonics without introducing ringing artifacts.
    *   **AI Noise Reduction**: Utilizes stationary noise profiling to eliminate background tape hiss and white noise.
    *   **Voice Isolation**: Applies a Bandpass Filter (300Hz - 3400Hz) to isolate human speech frequencies.
    *   **Extreme Voice Boost**: Amplifies the remaining voice signals to make them highly audible.

*   **Forensic Audio Visualization (`02forensics.py`)**:
    *   **1D FFT Spectrum**: Compares the frequency magnitude of raw and cleaned audio to visualize the removal of jamming peaks.
    *   **2D Spectrograms**: Shows Time vs. Frequency heatmaps of both audio states.
    *   **3D FFT Waterfall Plot**: Generates a 3D topographic view of the frequency spectrum over time, clearly showing "jamming walls" in the raw audio and their absence in the cleaned version.

*   **Automated Transcription (`03transcribe.py`)**:
    *   Uses **OpenAI's Whisper AI (Small Model)** to accurately transcribe the cleaned audio file.
    *   Tuned with specific beam size, temperature fallbacks, and logprob thresholds for highly corrupted audio.

## Prerequisites

Make sure you have Python 3.8+ installed. 
Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

*(Note: The transcription step requires `openai-whisper` and may require FFmpeg to be installed on your system).*

## Usage Workflow

The pipeline is split into three sequential scripts:

### Step 1: Clean the Audio
Place your corrupted audio file as `war-intercept.wav` in the root directory and run:
```bash
python 01audiocleanup.py
```
*Output: `01warintercept.wav` (The cleaned audio file)*

### Step 2: Generate Forensic Visuals
Analyze the difference between the raw and cleaned audio:
```bash
python 02forensics.py
```
*Outputs: `02fig1_fft.png`, `02fig2_spectrogram.png`, `02fig3_3D_FFT.png`*

### Step 3: Transcribe the Audio
Extract the dialogue from the cleaned audio file:
```bash
python 03transcribe.py
```
*Output: `03transcript.txt` (The transcribed text)*

## Project Structure

*   `01audiocleanup.py`: Script for noise reduction and frequency filtering.
*   `02forensics.py`: Script for generating 1D, 2D, and 3D spectral visualizations.
*   `03transcribe.py`: Script for AI-driven transcription.
*   `requirements.txt`: Python package dependencies.
*   `war-intercept.wav`: The original corrupted audio file (input).
*   `01warintercept.wav`: The processed, clean audio file (output).
*   `03transcript.txt`: The final transcription (output).
*   `*.png`: Generated frequency and spectrogram charts.
*   `Challenge3_Hrichik_Khandait.pdf` & `Challenge3_Submission_Report.docx`: Detailed submission and methodology reports.

## License
This project is provided for educational and forensics analysis purposes.
