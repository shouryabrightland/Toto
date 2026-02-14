import torch
import numpy as np
import sounddevice as sd
from API.modules.AudioEngine import AudioEngine

# ------------------ Load Silero VAD ------------------
vad_model, _ = torch.hub.load(
    repo_or_dir="models/snakers4_silero-vad_master",
    model="silero_vad",
    source="local"
)
vad_model.eval()

# ------------------ Config ------------------
SR = 16000
CHUNK = 512                     # ~32 ms
PRE_ROLL_SEC = 0.5
MAX_SILENCE_SEC = 0.35          # middle gap compression
END_SILENCE_SEC = 1.3           # stop condition
MAX_AUDIO_SEC = 15

PRE_ROLL_SAMPLES = int(PRE_ROLL_SEC * SR)
MAX_GAP_CHUNKS = int(MAX_SILENCE_SEC * SR / CHUNK)
END_SILENCE_CHUNKS = int(END_SILENCE_SEC * SR / CHUNK)

ENERGY_FLOOR = 1e-4
ENERGY_RATIO = 2.5
VAD_THRESH = 0.7

# ------------------ Listener ------------------
def listen_for_voice(speaker: AudioEngine):
    speaker.play_tts_file("effects/mic.mp3", volume=1)

    pre_roll = []
    output_chunks = []
    silence_buffer = []

    env_buffer = []
    PAST_ENV_SIZE = 400

    started = False
    end_silence = 0

    with sd.InputStream(
        samplerate=SR,
        channels=1,
        blocksize=CHUNK,
        dtype="float32"
    ) as stream:

        while True:
            audio, _ = stream.read(CHUNK)
            audio = np.squeeze(audio)

            # -------- Energy --------
            rms = np.sqrt(np.mean(audio ** 2))
            env_buffer.append(rms)
            if len(env_buffer) > PAST_ENV_SIZE:
                env_buffer.pop(0)

            noise_floor = max(np.mean(env_buffer), ENERGY_FLOOR)
            energy_hit = rms > noise_floor * ENERGY_RATIO

            # -------- Pre-roll --------
            if not started:
                pre_roll.extend(audio)
                if len(pre_roll) > PRE_ROLL_SAMPLES:
                    pre_roll = pre_roll[-PRE_ROLL_SAMPLES:]

            # -------- Fast reject --------
            vad_hit = False
            if energy_hit:
                prob = vad_model(
                    torch.from_numpy(audio).unsqueeze(0),
                    SR
                ).item()
                vad_hit = prob > VAD_THRESH

            is_speech = energy_hit or vad_hit

            # -------- Logic --------
            if is_speech:
                if not started:
                    started = True
                    output_chunks.append(np.array(pre_roll))
                    pre_roll.clear()

                # flush compressed silence
                if silence_buffer:
                    output_chunks.extend(silence_buffer[:MAX_GAP_CHUNKS])
                    silence_buffer.clear()

                output_chunks.append(audio)
                end_silence = 0

            else:
                if started:
                    silence_buffer.append(audio)
                    end_silence += 1

            # -------- Stop conditions --------
            if started and end_silence > END_SILENCE_CHUNKS:
                break

            if started and len(output_chunks) * CHUNK > SR * MAX_AUDIO_SEC:
                print("[warning] max audio length reached")
                break

    if not output_chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(output_chunks).astype(np.float32)

