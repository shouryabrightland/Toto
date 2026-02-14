from piper.voice import PiperVoice
import numpy as np
import threading
import queue
import time

from API.modules.AudioEngine import AudioEngine

class PiperTTS:
    def __init__(self, model_path, speaker: AudioEngine):
        if not isinstance(model_path, str):
            raise TypeError("model_path must be a string")
        
        print("loading Piper voice...", model_path)
        self.voice = PiperVoice.load(model_path)
        print("loaded", model_path)

        syn = self.voice.config
        syn.length_scale = 1.2
        
        self.SR = speaker.SR
        self.speaker = speaker

        # Queue and worker thread for non-blocking TTS
        self.q = queue.Queue()
        self.running = True
        #self.speaking = False  # Indicates if currently speaking
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    # ---------------------------
    # Public API
    # ---------------------------
    def enqueue(self, text: str,slow = None):
        """Add text to TTS queue for non-blocking speech"""
        self.q.put((text,slow))

    def stop(self):
        """Stop current speaking"""
        self.speaker.stop_bg()  # Stop background or current playback
        #self.speaking = False

    def shutdown(self):
        """Stop the TTS worker"""
        self.running = False
        self.q.put(None)

    # ---------------------------
    # Internal worker
    # ---------------------------
    def _worker(self):
        
        while True:
            text, slow = self.q.get()
            syn = self.voice.config
            if slow:
                syn.length_scale = 1.8  # ~10–15% slower
            else:
                syn.length_scale = 1.2

            for samples in self.synthesize_stream(text):
                self.speaker.play_samples(samples)

            self.q.task_done()

    # ---------------------------
    # Streaming synthesis
    # ---------------------------
    PAUSE_MAP = {
        ".": ".     ",   # long stop
        "?": "?     ",
        "!": "!     ",
        ",": ",  ",      # short breath
        ":": ":   ",
        ";": ";   "
    }
    def expand_punctuation(text: str) -> str:
        for p, repl in PiperTTS.PAUSE_MAP.items():
            text = text.replace(p, repl)
        return text

    def synthesize_stream(self, text, syn_config=None):
        for chunk in self.voice.synthesize(text, syn_config=syn_config):
            samples = chunk.audio_float_array
            if samples is not None:
                yield np.asarray(samples, dtype=np.float32)






























# from piper.voice import PiperVoice
# import numpy as np
# import time

# from API.modules.AudioEngine import AudioEngine

# class PiperTTS:
#     def __init__(self, model_path,speaker:AudioEngine):
#         if not isinstance(model_path, str):
#             raise TypeError("model_path must be a string")
#         print("loading Piper voice...", model_path)
#         self.voice = PiperVoice.load(model_path)
#         print("loaded", model_path)
#         self.SR = speaker.SR
#         self.speaker = speaker

#     def speak(self,text):
#         for samples in self.synthesize_stream(text):
#             self.speaker.play_samples(samples)

#     def synthesize_stream(self, text):
#         CROSSFADE_SAMPLES = self.SR
#         DELAY = 0.055

#         prev_tail = None

#         for chunk in self.voice.synthesize(text):
#             samples = chunk.audio_float_array
#             if samples is None:
#                 continue

#             samples = np.asarray(samples, dtype=np.float32)

#             if prev_tail is not None:
#                 fade_len = min(CROSSFADE_SAMPLES, len(samples), len(prev_tail))
#                 if fade_len > 0:
#                     fade_in = np.linspace(0.0, 1.0, fade_len)
#                     fade_out = 1.0 - fade_in
#                     samples[:fade_len] = (
#                         samples[:fade_len] * fade_in +
#                         prev_tail[-fade_len:] * fade_out
#                     )

#             prev_tail = samples[-CROSSFADE_SAMPLES:].copy()
#             yield samples
