import time
import soundfile as sf
import sounddevice as sd
from faster_whisper import WhisperModel
# ---------------------------
# Audio recording function
# ---------------------------
def record_audio(file_path="input.wav", duration=5, samplerate=16000):
    """
    Records audio from microphone and saves as WAV
    """
    print(f"Recording {duration} seconds...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()  # wait until recording is finished
    sf.write(file_path, audio_data, samplerate)
    print(f"Saved recording to {file_path}")
    return file_path

# ---------------------------
# Whisper API class
# ---------------------------
class WhisperAPI:
    def __init__(self, model_name="tiny.en"):
        """
        model_name: "tiny", "small", "medium", "large" (tiny is best for Pi)
        """
        
        print("Loading Whisper "+model_name+" model...")
        self.modelname = model_name
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8"
        )
        print("loaded whisper Model successfully...")
        
        self.recording_file = "input.wav"

    def transcribe(self, audio_path):
        """
        Transcribes audio file to text
        """
        #sf.write(self.recording_file, audio, self.SR)
        #------------------------------
        start = time.perf_counter()
        segments, _ = self.model.transcribe(audio_path)
        text = " ".join([segment.text for segment in segments])
        end = time.perf_counter()
        print(end - start,"-s---------------------------")
        return text.strip()
    #---------------------------------------
#--------------------------------------------------------------------------------------


import numpy as np
import soundfile as sf

def highpass_fft(data, sr, cutoff=90):
    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1/sr)
    fft[freqs < cutoff] = 0
    return np.fft.irfft(fft, len(data))

def presence_boost(data, sr, freq=3500, width=600, gain_db=3):
    gain = 10 ** (gain_db / 20)
    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), 1/sr)

    band = (freqs > freq - width) & (freqs < freq + width)
    fft[band] *= gain

    return np.fft.irfft(fft, len(data))

def rms_normalize(data, target_db=-20):
    rms = np.sqrt(np.mean(data**2) + 1e-9)
    target_rms = 10 ** (target_db / 20)
    data *= target_rms / rms
    return np.clip(data, -1.0, 1.0)

def preprocess_audio(input_wav, output_wav):
    data, sr = sf.read(input_wav)
    if data.ndim > 1:
        data = data.mean(axis=1)

    data = highpass_fft(data, sr)
    data = presence_boost(data, sr)
    data = rms_normalize(data)

    sf.write(output_wav, data, sr)

whisT = WhisperAPI("tiny.en")
whisM = WhisperAPI("medium.en")
whisS = WhisperAPI("small.en")
whisB = WhisperAPI("base.en")
c = "samples/"+input("enter name")+".wav"
lis = [whisT,whisB,whisS,whisM]
#record_audio(c)
#preprocess_audio(c,c+".clear.mp3")
for m in lis:
    print(m.modelname)
    print(m.transcribe(c+".clear.wav"))
    print(m.transcribe(c))


#xx 0.8 -- 0.6 -- 1.8 -- 5.0
#x- 