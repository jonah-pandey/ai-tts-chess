import os

import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import requests
import json
import pyttsx3

def tts_test():
    '''tests tts functionality'''
    engine = pyttsx3.init()

    engine.say("test")
    engine.runAndWait()

#flag for debug outputs
debug = True
name = 'jonah'

os.chdir(f'C:\\Users\\{name}\\Downloads\\')

def tts(speech):
    engine = pyttsx3.init()
    engine.say(speech)
    engine.runAndWait()



def record(duration=5, path=f"C:\\Users\\{name}\\Downloads\\output-mic.wav"):
    '''record single channel audio into a wav file'''
    fs = 44100  # Sample rate (44.1 kHz is CD quality)

    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Done.")

    write(path, fs, audio)  # Save as WAV file

    if debug == True:
        print(f"file exists: {os.path.exists(path)}")


def transcribe(path=f"C:\\Users\\{name}\\Downloads\\output-mic.wav"):
    '''transcribe a wav file with Whisper'''
    model = whisper.load_model("base")
    result = model.transcribe(path)
    return result["text"]

def test():
    '''test the main loop'''
    record()
    print(transcribe())

def test_llm():
    '''records a message, parses to whisper and then returns with tts'''
    record()

    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama-3.2-1b-instruct",
        "messages": [{"role": "user", "content": transcribe()}],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    print(response.json()['choices'][0]['message']['content'])
    tts((response.json()['choices'][0]['message']['content']))



test_llm()
# test()


