import os

import sounddevice as sd
from scipy.io.wavfile import write
import whisper

debug = True

os.chdir('C:\\Users\\jonah\\Downloads\\')


def record(duration=5, path="C:\\Users\\jonah\\Downloads\\output-mic.wav"):
    '''record single channel audio into a wav file'''
    fs = 44100  # Sample rate (44.1 kHz is CD quality)

    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Done.")

    write(path, fs, audio)  # Save as WAV file

    if debug == True:
        print(f"file exists: {os.path.exists(path)}")


def transcribe(path="C:\\Users\\jonah\\Downloads\\output-mic.wav"):
    '''transcribe a wav file with Whisper'''
    model = whisper.load_model("tiny")
    result = model.transcribe(path)
    return "\n"+result["text"]

def test():
    '''test the main loop'''
    record()
    print(transcribe())



