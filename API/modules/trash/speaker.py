import sounddevice as sd
import soundfile as sf

class Speaker:
    def __init__(self,samplerate = 16000):
        print("opening output stream...")
        self.stream = sd.OutputStream(samplerate=samplerate, channels=1, dtype="float32")
        self.stream.start()
        self.SR = samplerate
        print("opened output stream...")
    
    def play(self, file_path, volume=1.0,wait=True):
        data, samplerate = sf.read(file_path, dtype="float32")
        # safety clamp
        volume = max(0.0, min(volume, 1.0))
        data *= volume
        sd.play(data, samplerate)
        if wait:
            sd.wait()
    
    