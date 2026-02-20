import numpy as np
import sounddevice as sd
import onnxruntime as ort
import time
import threading
from collections import deque
from queue import Queue

from API.modules.Logging import Log
from API.whisper_api import WhisperAPI


class WakeAssistant:

    # ---------------- INIT ----------------
    def __init__(
        self,
        model_path: str,
        whisper_api: WhisperAPI,
        speaker=None,
        onDetect=None,
        onStopRecording=None,
    ):

        self.log = Log("Recorder").log

        # -------- CONFIG --------
        self.SR = 16000
        self.WINDOW_SEC = 1.5
        self.HOP_SEC = 0.1

        self.THRESHOLD = 0.6
        self.VOTE_WINDOW = 5
        self.VOTE_REQUIRED = 3
        self.COOLDOWN_SEC = 1.0

        # -------- Adaptive Silence --------
        self.SILENCE_DURATION = 1.0
        self.MIN_RECORDING_SEC = 3
        self.PRE_ROLL_SEC = 0.3

        self.NOISE_ALPHA = 0.95        # noise smoothing
        # self.SPEECH_MULTIPLIER = 1.5   # speech threshold = noise * multiplier
        self.MIN_THRESHOLD = 0.008     # safety floor

        # -------- Auto Gain --------
        self.TARGET_RMS = 0.05
        self.MAX_GAIN = 10.0
        self.gain = 1.0

        # -------- Derived --------
        self.SAMPLES = int(self.SR * self.WINDOW_SEC)
        self.HOP_SAMPLES = int(self.SR * self.HOP_SEC)
        self.max_silence_samples = int(self.SR * self.SILENCE_DURATION)
        self.min_recording_samples = int(self.SR * self.MIN_RECORDING_SEC)

        # -------- Load Wake Model --------
        self.log("Loading WakeupWord model..", model_path)

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.log("✅ ONNX model loaded:", model_path)

        # -------- State --------
        self.audio_buffer = deque(maxlen=self.SAMPLES)
        self.score_buffer = deque(maxlen=self.VOTE_WINDOW)
        self.pre_roll = deque(maxlen=int(self.PRE_ROLL_SEC * self.SR))

        self.last_trigger = 0.0
        self.recording = False
        self.recorded_audio = []

        self.silence_counter = 0
        self.total_recorded = 0
        self.noise_floor = 0.01

        # -------- External --------
        self.command_queue = Queue()
        self.whisper = whisper_api
        self.speaker = speaker
        self.onDetect = onDetect
        self.onStopRecording = onStopRecording

    # ---------------- UTILS ----------------
    def compute_rms(self, chunk):
        return np.sqrt(np.mean(chunk ** 2) + 1e-8)

    def auto_gain(self, chunk):
        rms = self.compute_rms(chunk)
        if rms > 0:
            desired_gain = self.TARGET_RMS / rms
            desired_gain = min(desired_gain, self.MAX_GAIN)
            self.gain = 0.9 * self.gain + 0.1 * desired_gain
        return np.clip(chunk * self.gain, -1.0, 1.0)

    def update_noise_floor(self, energy):
        self.noise_floor = (
            self.NOISE_ALPHA * self.noise_floor
            + (1 - self.NOISE_ALPHA) * energy
        )

    # ---------------- ASSISTANT WORKER ----------------
    def assistant_worker(self, output_queue: Queue):
        self.log("Transcriber started")

        while True:
            audio = self.command_queue.get()
            if audio is None:
                continue

            self.log("Transcribing audio", audio.shape)
            text = self.whisper.transcribe(audio)
            self.log("🗣 User:", text)
            output_queue.put(text)

    # ---------------- WAKE DETECTION ----------------
    def wake_detection(self):

        audio = np.array(self.audio_buffer, dtype=np.float32)
        audio = audio / (np.max(np.abs(audio)) + 1e-6)

        x = audio[np.newaxis, ..., np.newaxis].astype(np.float32)
        outputs = self.session.run([self.output_name], {self.input_name: x})
        score = float(outputs[0][0][0])

        self.score_buffer.append(score)
        positives = sum(s >= self.THRESHOLD for s in self.score_buffer)

        now = time.time()

        if positives >= self.VOTE_REQUIRED and (now - self.last_trigger) > self.COOLDOWN_SEC:

            self.log(f"\n🔥 WAKE WORD DETECTED | score={score:.3f}")

            if self.speaker:
                self.speaker.stop_bg()
                self.speaker.play_file("effects/mic.mp3", volume=1)

            if self.onDetect:
                self.onDetect()

            self.last_trigger = now
            self.score_buffer.clear()

            self.recording = True
            self.recorded_audio = list(self.pre_roll)
            self.silence_counter = 0
            self.total_recorded = len(self.recorded_audio)

        else:
            print(f"\rscore={score:.3f} | votes={positives}/{self.VOTE_WINDOW}", end="")

    # ---------------- RECORD COMMAND ----------------
    def process_recording(self, chunk):

        chunk = chunk.astype(np.float32)
        chunk = self.auto_gain(chunk)

        energy = self.compute_rms(chunk)

        dynamic_threshold = max(self.noise_floor, self.MIN_THRESHOLD)
        speech_ratio = energy / (dynamic_threshold + 1e-6)

        self.recorded_audio.extend(chunk)
        self.total_recorded += len(chunk)
        #self.log("speech",dynamic_threshold,energy)
        if speech_ratio < 1.4:
            self.silence_counter += len(chunk)
        else:
            self.silence_counter = 0
            self.log("speech",dynamic_threshold,energy)

        if (
            self.silence_counter > self.max_silence_samples
            and self.total_recorded > self.min_recording_samples
        ):

            self.log(
                f"🛑 Speech ended | noise={self.noise_floor:.4f} "
                f"| threshold={dynamic_threshold:.4f}"
            )

            self.recording = False

            if self.onStopRecording:
                self.onStopRecording()

            audio_np = np.array(self.recorded_audio, dtype=np.float32)

            # reset
            self.recorded_audio = []
            self.silence_counter = 0
            self.total_recorded = 0

            self.command_queue.put(audio_np)

            if self.speaker:
                self.speaker.play_file("effects/recordend.mp3", volume=1)

    # ---------------- AUDIO CALLBACK ----------------
    def audio_callback(self, indata, frames, time_info, status):

        if status:
            self.log("Audio status:", status)

        chunk = indata[:, 0]

        self.audio_buffer.extend(chunk)
        self.pre_roll.extend(chunk)
        

        if not self.recording:
            if len(self.audio_buffer) == self.SAMPLES:
                energy = self.compute_rms(chunk)
                self.update_noise_floor(energy)
                self.wake_detection()
        else:
            self.process_recording(chunk)

    # ---------------- START ----------------
    def start(self, output_queue: Queue):

        threading.Thread(
            target=self.assistant_worker,
            daemon=True,
            args=(output_queue,),
        ).start()

        self.log("All set.. Requesting Audio Stream")

        try:
            with sd.InputStream(
                samplerate=self.SR,
                channels=1,
                blocksize=self.HOP_SAMPLES,
                dtype="float32",
                callback=self.audio_callback,
            ):
                while True:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            self.log("🛑 Stopped by User")
