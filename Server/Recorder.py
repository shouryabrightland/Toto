import numpy as np
import sounddevice as sd
import onnxruntime as ort
import time
import webrtcvad
import threading
from collections import deque
from queue import Queue

from API.whisper_api import WhisperAPI

class WakeAssistant:

    # ---------------- INIT ----------------
    def __init__(self, model_path: str, whisper_api:WhisperAPI):

        # -------- CONFIG --------
        self.SR = 16000
        self.WINDOW_SEC = 1.5
        self.HOP_SEC = 0.1

        self.THRESHOLD = 0.6
        self.VOTE_WINDOW = 5
        self.VOTE_REQUIRED = 3
        self.COOLDOWN_SEC = 1.0

        self.VAD_MODE = 2
        self.FRAME_MS = 30
        self.SILENCE_DURATION = 1.5
        self.PRE_ROLL_SEC = 0.3

        # -------- Derived --------
        self.SAMPLES = int(self.SR * self.WINDOW_SEC)
        self.HOP_SAMPLES = int(self.SR * self.HOP_SEC)
        self.FRAME_SIZE = int(self.SR * self.FRAME_MS / 1000)
        self.max_silence_frames = int(self.SILENCE_DURATION * 1000 / self.FRAME_MS)

        # -------- Model --------
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print("✅ ONNX model loaded:", model_path)

        # -------- State --------
        self.audio_buffer = deque(maxlen=self.SAMPLES)
        self.score_buffer = deque(maxlen=self.VOTE_WINDOW)
        self.pre_roll = deque(maxlen=int(self.PRE_ROLL_SEC * self.SR))

        self.vad = webrtcvad.Vad(self.VAD_MODE)

        self.last_trigger = 0.0
        self.recording = False
        self.recorded_audio = []
        self.silence_frames = 0

        # -------- External --------
        self.command_queue = Queue()
        self.whisper = whisper_api
        self.speaker = whisper_api.speaker


    # ---------------- UTILS ----------------
    def peak_norm(self, chunk):
        p = np.max(np.abs(chunk))
        return chunk if p == 0 else chunk / p


    # ---------------- ASSISTANT WORKER ----------------
    def assistant_worker(self, output_queue: Queue):

        print("🧵 Assistant worker started")

        while True:
            audio = self.command_queue.get()

            print("\n🧠 Processing command...")

            text = self.whisper.transcribe(audio)
            print("🗣 User:", text)

            output_queue.put(text)


    # ---------------- WAKE DETECTION ----------------
    def wake_detection(self):

        audio = np.array(self.audio_buffer, dtype=np.float32)
        audio = self.peak_norm(audio)

        x = audio[np.newaxis, ..., np.newaxis].astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: x})
        score = float(outputs[0][0][0])

        self.score_buffer.append(score)
        positives = sum(s >= self.THRESHOLD for s in self.score_buffer)

        now = time.time()

        if positives >= self.VOTE_REQUIRED and (now - self.last_trigger) > self.COOLDOWN_SEC:

            print(f"\n🔥 WAKE WORD DETECTED | score={score:.3f}")

            if self.speaker:
                self.speaker.stop_bg() #start of mic
                self.speaker.play_file("effects/mic.mp3", volume=1)

            self.last_trigger = now
            self.score_buffer.clear()

            self.recording = True
            self.recorded_audio = list(self.pre_roll)
            self.silence_frames = 0

        else:
            print(f"\rscore={score:.3f} | votes={positives}/{self.VOTE_WINDOW}", end="")


    # ---------------- RECORD COMMAND ----------------
    def process_recording(self, chunk):

        frame_int16 = (chunk * 32768).astype(np.int16)

        for i in range(0, len(frame_int16), self.FRAME_SIZE):

            frame = frame_int16[i:i + self.FRAME_SIZE]
            if len(frame) < self.FRAME_SIZE:
                continue

            is_speech = self.vad.is_speech(frame.tobytes(), self.SR)
            self.recorded_audio.extend(frame)

            if is_speech:
                self.silence_frames = 0
            else:
                self.silence_frames += 1

            if self.silence_frames > self.max_silence_frames:

                print("🛑 Speech ended")

                self.recording = False

                audio_np = np.array(self.recorded_audio, dtype=np.int16)
                audio_np = audio_np.astype(np.float32) / 32768.0

                self.command_queue.put(audio_np)

                if self.speaker:
                    self.speaker.play_file("effects/recordend.mp3", volume=1)

                break


    # ---------------- AUDIO CALLBACK ----------------
    def audio_callback(self, indata, frames, time_info, status):

        chunk = indata[:, 0]

        self.audio_buffer.extend(chunk)
        self.pre_roll.extend((chunk * 32768).astype(np.int16))

        if not self.recording:

            if len(self.audio_buffer) == self.SAMPLES:
                self.wake_detection()

        else:
            self.process_recording(chunk)


    # ---------------- START ----------------
    def start(self, output_queue: Queue):

        print("🎙️ Listening (Class Architecture)...")

        threading.Thread(
            target=self.assistant_worker,
            daemon=True,
            args=(output_queue,)
        ).start()

        try:
            with sd.InputStream(
                samplerate=self.SR,
                channels=1,
                blocksize=self.HOP_SAMPLES,
                dtype="float32",
                callback=self.audio_callback
            ):
                while True:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n🛑 Stopped")
