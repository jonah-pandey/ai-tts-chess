import os

import sounddevice as sd
from scipy.io.wavfile import write
import whisper
os.chdir('C:\\Users\\jonah\\Downloads\\')

fs = 44100  # Sample rate (44.1 kHz is CD quality)
duration = 5  # Duration in seconds

print("Recording...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()  # Wait until recording is finished
print("Done.")

write("output-mic.wav", fs, audio)  # Save as WAV file
print(f"file exists: {os.path.exists("C:\\Users\\jonah\\Downloads\\output-mic.wav")}")


# transcribe with Whisper
model = whisper.load_model("base")
result = model.transcribe("C:\\Users\\jonah\\Downloads\\output-mic.wav")
print(result["text"])