import threading
import queue
import sounddevice as sd
import soundfile as sf
import numpy as np


class AudioEngine:
    def __init__(self, samplerate=16000, blocksize=1024):
        self.SR = samplerate
        self.BLOCK = blocksize

        self.q = queue.Queue()
        self.running = True

        # Thread-safe stop signal for background audio
        self.bg_stop_event = threading.Event()

        self.bgAllow = True

        # Output stream
        self.stream = sd.OutputStream(
            samplerate=self.SR,
            channels=1,
            dtype="float32",
            blocksize=self.BLOCK
        )
        self.stream.start()

        # Worker thread
        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )
        self.thread.start()

    # ─────────────────────────────
    # PUBLIC API
    # ─────────────────────────────

    def play_bg_file(self, file_path, volume=0.4):
        """Looping background sound (thinking / listening)"""
        self.bgAllow = True
        self.q.put(("bg", file_path, volume))

    def play_tts_file(self, file_path, volume=1.0):
        self.stop_bg()
        """Foreground TTS (preempts BG)"""
        self.q.put(("tts", file_path, volume))

    def play_samples(self, samples: np.ndarray):
        """Raw samples playback (preempts BG)"""
        self.stop_bg()
        self.q.put(("samples", samples, None))

    def stop_bg(self):
        """Stop background audio immediately"""
        self.bgAllow = False
        self.bg_stop_event.set()

    def shutdown(self):
        """Clean shutdown"""
        self.running = False
        self.bg_stop_event.set()
        self.q.put(("exit", None, None))

    # ─────────────────────────────
    # WORKER LOOP
    # ─────────────────────────────

    def _run(self):
        while self.running:
            job, a, b = self.q.get()

            if job == "exit":
                break

            if job == "bg" and self.bgAllow:
                self._play_bg_loop(a, b)

            elif job == "tts":
                self._play_tts(a, b)

            elif job == "samples":
                self._play_samples(a)

            self.q.task_done()

        self.stream.stop()
        self.stream.close()

    # ─────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────

    def _play_bg_loop(self, file_path, volume):
        data, sr = sf.read(file_path, dtype="float32")

        if data.ndim > 1:
            data = data.mean(axis=1)

        if sr != self.SR:
            data = self._resample(data, sr)

        data *= volume
        length = len(data)

        self.bg_stop_event.clear()

        while not self.bg_stop_event.is_set():
            for i in range(0, length, self.BLOCK):
                
                if self.bg_stop_event.is_set():
                    return
                self.stream.write(data[i:i + self.BLOCK])

    def _play_tts(self, file_path, volume):
        self.bg_stop_event.set()

        data, sr = sf.read(file_path, dtype="float32")

        if data.ndim > 1:
            data = data.mean(axis=1)

        if sr != self.SR:
            data = self._resample(data, sr)

        self.stream.write(data * volume)

    def _play_samples(self, samples):
        self.bg_stop_event.set()
        #print(self.bg_stop_event.is_set(),"o")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        self.stream.write(samples.astype(np.float32))

    def _resample(self, data, sr):
        ratio = self.SR / sr
        x_old = np.arange(len(data))
        x_new = np.linspace(0, len(data) - 1, int(len(data) * ratio))
        return np.interp(x_new, x_old, data).astype(np.float32)

































# import threading
# import queue
# import sounddevice as sd
# import soundfile as sf
# import numpy as np

# class AudioEngine:
#     def __init__(self, samplerate=16000):
#         self.SR = samplerate
#         self.q = queue.Queue()
#         self.running = True

#         self.bg_stop_flag = False

#         print("opening output stream...")
#         self.stream = sd.OutputStream(
#             samplerate=self.SR,
#             channels=1,
#             dtype="float32",
#             blocksize=1024
#         )
#         self.stream.start()
#         print("opened output stream...")

#         self.thread = threading.Thread(
#             target=self._run,
#             daemon=True
#         )
#         self.thread.start()

#     # ─────────────────────────────
#     # PUBLIC API
#     # ─────────────────────────────

#     def play_bg_file(self, file_path, volume=0.6):
#         self.q.put(("bg_file", file_path, volume))

#     def play_tts_file(self, file_path, volume=1.0):
#         self.stop_bg()
#         self.q.put(("tts_file", file_path, volume))

#     def play_samples(self, samples: np.ndarray):
#         self.stop_bg()
#         self.q.put(("samples", samples, None))

#     def stop_bg(self):
#         self.bg_stop_flag = True
#         sd.stop()

#     def shutdown(self):
#         self.running = False
#         self.q.put(("exit", None, None))

#     # ─────────────────────────────
#     # WORKER LOOP
#     # ─────────────────────────────

#     def _run(self):
#         while self.running:
#             job, a, b = self.q.get()

#             if job == "exit":
#                 self.q.task_done()
#                 break

#             if job == "bg_file":
#                 self._play_bg_file(a, b)

#             elif job == "tts_file":
#                 self._play_tts_file(a, b)

#             elif job == "samples":
#                 self._play_samples(a)
#             self.q.task_done()
            

#     # ─────────────────────────────
#     # INTERNAL HELPERS
#     # ─────────────────────────────

#     def _play_bg_file(self, file_path, volume):
#         data, sr = sf.read(file_path, dtype="float32")
#         if sr != self.SR:
#             data = self._resample(data, sr)

#         data *= volume
#         self.bg_stop_flag = False

#         chunk = 1024
#         for i in range(0, len(data), chunk):
#             if self.bg_stop_flag:
#                 break
#             self.stream.write(data[i:i+chunk])

#     def _play_tts_file(self, file_path, volume):
#         self.bg_stop_flag = True
#         sd.stop()

#         data, sr = sf.read(file_path, dtype="float32")
#         if sr != self.SR:
#             data = self._resample(data, sr)

#         self.stream.write(data * volume)

#     def _play_samples(self, samples):
#         self.bg_stop_flag = True
#         self.stream.write(samples)

#     def _resample(self, data, sr):
#         if data.ndim > 1:
#             data = data.mean(axis=1)

#         ratio = self.SR / sr
#         x_old = np.arange(len(data))
#         x_new = np.linspace(0, len(data) - 1, int(len(data) * ratio))
#         return np.interp(x_new, x_old, data).astype(np.float32)
