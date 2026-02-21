import sounddevice as sd
import numpy as np

DEVICE_INDEX = 2   # change if needed
DURATION = 10       # seconds
SAMPLE_RATE = 16000

print("Recording...")

recording = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='int16',
    device="pipewire"
)

sd.wait()

print("Recording complete. Playing back...")

sd.play(recording, SAMPLE_RATE)
sd.wait()

print("Done.")