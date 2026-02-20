import time
import soundfile as sf
import sounddevice as sd
from faster_whisper import WhisperModel
from API.modules.AudioEngine import AudioEngine
from API.modules.Logging import Log
# ---------------------------
# Whisper API class
# ---------------------------
class WhisperAPI:
    def __init__(self, model_name="base.en"):
        """
        model_name: "tiny", "small", "medium", "large" (tiny is best for Pi)
        """
        self.log = Log("WhisperAPI").log
        self.log("Loading Whisper",model_name,"model...")
        self.model = WhisperModel(
            "base.en",
            device="cpu",
            compute_type="int8"
        )
        self.log("loaded Model successfully...")
        self.recording_file = "input.wav"
    #---------------------------------------

    def transcribe(self,audio):
        """
         immediately transcribes audio
        """
        audio = audio.astype("float32")
        #time.sleep(5)---------------------
        t1 = time.perf_counter()
        self.log("transcripting Audio...")
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
        self.log("Transcribed:", text)
        t2 = time.perf_counter()
        self.log("time taken in transcription",(t2-t1)*1000,"ms")
        return text