import time
import soundfile as sf
import sounddevice as sd
from faster_whisper import WhisperModel
from API.modules.AudioEngine import AudioEngine
import API.modules.voice_detect as detector
# ---------------------------
# Audio recording function
# ---------------------------
# def record_audio(file_path="input.wav", duration=5, samplerate=16000):
#     """
#     Records audio from microphone and saves as WAV
#     """
#     print(f"Recording {duration} seconds...")
#     audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
#     sd.wait()  # wait until recording is finished
#     sf.write(file_path, audio_data, samplerate)
#     print(f"Saved recording to {file_path}")
#     return file_path

# ---------------------------
# Whisper API class
# ---------------------------
class WhisperAPI:
    def __init__(self,speaker:AudioEngine, model_name="base.en"):
        """
        model_name: "tiny", "small", "medium", "large" (tiny is best for Pi)
        """
        self.SR = speaker.SR
        print("Loading Whisper",model_name,"model...")
        self.model = WhisperModel(
            "base.en",
            device="cpu",
            compute_type="int8"
        )
        print("loaded whisper Model successfully...")
        self.speaker = speaker
        self.detector = detector
        self.recording_file = "input.wav"

    def transcribe(self, audio_path):
        """
        Transcribes audio file to text
        """
        #sf.write(self.recording_file, audio, self.SR)
        #------------------------------

        segments, _ = self.model.transcribe(audio_path)
        text = " ".join([segment.text for segment in segments])
        return text.strip()
    #---------------------------------------

    def record_and_transcribe(self):
        """
        Records audio and immediately transcribes it
        """
        # i done this so that i can put custom message during debeg in this text var...
        text = ""
        while not text:
            print("initizalising voice to text----------------")
            audio = self.detector.listen_for_voice(self.speaker)
            audio = audio.astype("float32")
            self.speaker.play_bg_file("effects/think.mp3", volume=0.5)
            #time.sleep(5)---------------------
            t1 = time.perf_counter()
            segments, _ = self.model.transcribe(
                audio,
                language="en",
                task="transcribe",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=False
            )
            text = " ".join(s.text for s in segments).strip(" ")
        print("Transcribed:", text)
        t2 = time.perf_counter()
        print("time taken in transcription",(t2-t1)*1000,"ms")
        sf.write(self.recording_file, audio, self.SR)

        print('written...')
        return text
        
#--------------------------------------------------------------------------------------


# import numpy as np
# import soundfile as sf

# def highpass_fft(data, sr, cutoff=90):
#     fft = np.fft.rfft(data)
#     freqs = np.fft.rfftfreq(len(data), 1/sr)
#     fft[freqs < cutoff] = 0
#     return np.fft.irfft(fft, len(data))

# def presence_boost(data, sr, freq=3500, width=600, gain_db=3):
#     gain = 10 ** (gain_db / 20)
#     fft = np.fft.rfft(data)
#     freqs = np.fft.rfftfreq(len(data), 1/sr)

#     band = (freqs > freq - width) & (freqs < freq + width)
#     fft[band] *= gain

#     return np.fft.irfft(fft, len(data))

# def rms_normalize(data, target_db=-20):
#     rms = np.sqrt(np.mean(data**2) + 1e-9)
#     target_rms = 10 ** (target_db / 20)
#     data *= target_rms / rms
#     return np.clip(data, -1.0, 1.0)

# def preprocess_audio(input_wav, output_wav):
#     data, sr = sf.read(input_wav)
#     if data.ndim > 1:
#         data = data.mean(axis=1)

#     data = highpass_fft(data, sr)
#     data = presence_boost(data, sr)
#     data = rms_normalize(data)

#     sf.write(output_wav, data, sr)

# speaker = AudioEngine()
# whis = WhisperAPI(speaker)
# preprocess_audio("samples/convo.mp3","samples/convo_cleared.mp3")
# whis.transcribe("samples/convo_cleared.mp3")
