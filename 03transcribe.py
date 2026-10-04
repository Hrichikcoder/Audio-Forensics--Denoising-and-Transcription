import whisper
import json

print("Loading Whisper model (small)...")
model = whisper.load_model("small")

print("Transcribing 01warintercept.wav... (This takes a couple of minutes)")
result = model.transcribe(
    "01warintercept.wav",
    language='en',
    beam_size=10,
    best_of=10,
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    condition_on_previous_text=False,
    compression_ratio_threshold=2.4,
    logprob_threshold=-1.0,
    no_speech_threshold=0.6
)

with open("03transcript.txt", "w", encoding="utf-8") as f:
    f.write(result['text'].strip())

print("Done! Output saved as 03transcript.txt")