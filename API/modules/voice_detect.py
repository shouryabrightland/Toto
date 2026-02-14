import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
from API.modules.AudioEngine import AudioEngine
#from AudioEngine import AudioEngine
import time

# ------------------ Load Silero VAD ------------------

vad_model, _ = torch.hub.load(
    repo_or_dir="models/snakers4_silero-vad_master",
    model="silero_vad",
    source="local"
)
vad_model.eval()


# ------------------ Config ------------------
DEBUG = True

SR = 16000
CHUNK = 512                     # ~32 ms
PRE_ROLL_SEC = 0.7
MAX_SILENCE_SEC = 1          # middle gap compression
END_SILENCE_SEC = 2           # stop condition
MAX_AUDIO_SEC = 30

PRE_ROLL_SAMPLES = int(PRE_ROLL_SEC * SR)
MAX_GAP_CHUNKS = int(MAX_SILENCE_SEC * SR / CHUNK)
END_SILENCE_CHUNKS = int(END_SILENCE_SEC * SR / CHUNK)

ENERGY_FLOOR = 5e-3
ENERGY_RATIO = 2.5
VAD_THRESH = 0.7


# ------------------ Listener ------------------
def listen_for_voice(speaker: AudioEngine):
    #high pass
    hp_prev_x = 0.0
    hp_prev_y = 0.0
    HP_ALPHA = 0.995  # closer to 1 = lower cutoff (~80–100 Hz)

    if speaker:
        speaker.play_tts_file(file_path="effects/mic.mp3", volume=1)

    pre_roll = []
    output_chunks = []
    silence_buffer = []

    env_buffer = []
    PAST_ENV_SIZE = 150 #5sec

    started = False
    end_silence = 0

    

    with sd.InputStream(
        samplerate=SR,
        channels=1,
        blocksize=CHUNK,
        dtype="float32"
    ) as stream:
        CH_FRAME = -1
        total_samples = 0

        avg_time=0
        
        avg1 = 0
        avg2 = 0
        avg3 = 0
        avg4 = 0
        avg5 = 0
        avg6 = 0
        avg7 = 0

        while True:


            t1 = time.perf_counter()
            CH_FRAME +=1
            audio, _ = stream.read(CHUNK)
            t2 = time.perf_counter()
            audio = np.squeeze(audio)
            

            # -------- Energy --------
            rms = np.sqrt(np.mean(audio ** 2))
            env_buffer.append(rms)
            if len(env_buffer) > PAST_ENV_SIZE:
                env_buffer.pop(0)

            if not started:
                noise_floor = max(np.mean(env_buffer), ENERGY_FLOOR)
                energy_hit = rms > noise_floor * ENERGY_RATIO

            else:
               # update only during silence
                if not vad_hit and not energy_hit:
                    noise_floor = max(
                        0.995 * noise_floor + 0.005 * rms,
                        ENERGY_FLOOR
                    )
                energy_hit = rms > noise_floor * ENERGY_RATIO

            t3 = time.perf_counter()

            # -------- Pre-roll --------
            if not started:
                pre_roll.extend(audio)
                if len(pre_roll) > PRE_ROLL_SAMPLES:
                    pre_roll = pre_roll[-PRE_ROLL_SAMPLES:]

            # -------- Fast reject --------

            t4 = time.perf_counter()

            vad_hit = False
            if energy_hit and (started or CH_FRAME % 3 == 0):
                if DEBUG:
                    print("hit FLOOR=",noise_floor,"current=",rms)#
                prob = vad_model(
                    torch.from_numpy(audio).unsqueeze(0),
                    SR
                ).item()
                vad_hit = prob > VAD_THRESH

            is_speech = vad_hit or (energy_hit and started)
            chunk_raw = audio.copy()

            t5 = time.perf_counter()

            # high-pass
            audio, hp_prev_x, hp_prev_y = highpass_simple(
                audio, hp_prev_x, hp_prev_y, HP_ALPHA
            )

            t6 = time.perf_counter()

            # -------- Logic --------
            if is_speech:
                # presence
                audio = np.clip(audio * 1.3, -1.0, 1.0)


                if not started:
                    started = True
                    if speaker:
                        speaker.play_tts_file(file_path="effects/recording.mp3", volume=0.5)
                    print("Recording Started")

                    pre = np.array(pre_roll, dtype=np.float32)
                    pre, hp_prev_x, hp_prev_y = highpass_simple(
                        pre, hp_prev_x, hp_prev_y, HP_ALPHA
                    )

                    output_chunks.append(pre)
                    total_samples += len(pre)
                    del pre
                    pre_roll.clear()

                # flush compressed silence
                if silence_buffer:
                    sil = np.concatenate(silence_buffer)
                    sil, hp_prev_x, hp_prev_y = highpass_simple(sil, hp_prev_x, hp_prev_y, HP_ALPHA)

                    max_keep = MAX_GAP_CHUNKS * CHUNK

                    if len(sil) > max_keep:
                        keep = max_keep // 2
                        sil = np.concatenate([sil[:keep], sil[-keep:]])

                    output_chunks.append(sil)
                    total_samples += len(sil)
                    silence_buffer.clear()

                output_chunks.append(audio)
                total_samples += len(audio)
                end_silence = 0

            else:
                if started:
                    silence_buffer.append(chunk_raw)
                    end_silence += 1
            
            t7 = time.perf_counter()
            # -------- Stop conditions --------
            if started and end_silence > END_SILENCE_CHUNKS:
                print("recording end")

                # KEEP tail (last speech decay)
                tail_chunks = silence_buffer[:int(0.4 * SR / CHUNK)]
                if tail_chunks:
                    tail = np.concatenate(tail_chunks)
                    tail, hp_prev_x, hp_prev_y = highpass_simple(
                        tail, hp_prev_x, hp_prev_y, HP_ALPHA
                    )
                    output_chunks.append(tail)
                    total_samples += len(tail)

                silence_buffer.clear()
                if speaker:
                    speaker.play_tts_file(file_path="effects/recordend.mp3", volume=1)
                    #speaker.q.join()
                break

            if started and total_samples > SR * MAX_AUDIO_SEC:
                print("[warning] max audio length reached")
                break
            time.sleep(0.004)
            t8 = time.perf_counter()

            avg1 = (t2 - t1+avg1)/2
            avg2 = (t3 - t2+avg2)/2
            avg3 = (t4 - t3+avg3)/2
            avg4 = (t5 - t4+avg4)/2
            avg5 = (t6 - t5+avg5)/2
            avg6 = (t7 - t6+avg6)/2
            avg7 = (t8 - t7+avg7)/2
            avg_time = (t8-t2+avg_time)/2
        if DEBUG:
            print((avg1*1000),"ms per chunk","stream read")
            print((avg2*1000),"ms per chunk","energy hit")
            print((avg3*1000),"ms per chunk","pre-roll")
            print((avg4*1000),"ms per chunk","fast rejection vbd")
            print((avg5*1000),"ms per chunk","single high pass")
            print((avg6*1000),"ms per chunk","logic")
            print((avg7*1000),"ms per chunk","end")
            print("total",avg_time*1000)

    if not output_chunks:
        return np.array([], dtype=np.float32)

    return rms_normalize(np.concatenate(output_chunks).astype(np.float32))

def rms_normalize(x, target_db=-20):
    rms = np.sqrt(np.mean(x**2) + 1e-9)
    gain = (10 ** (target_db / 20)) / rms
    return np.clip(x * gain, -1.0, 1.0)

def highpass_simple(x, prev_x, prev_y, alpha):
    y = np.empty_like(x)
    for i in range(len(x)):
        y[i] = alpha * (prev_y + x[i] - prev_x)
        prev_x = x[i]
        prev_y = y[i]
    return y, prev_x, prev_y

# def record_audio(file_path="input.wav", duration=5, samplerate=16000):
#     """
#     Records audio from microphone and saves as WAV
#     """
#     print(f"Recording {duration} seconds...")
#     audio_data = listen_for_voice(AudioEngine(16000))
#     sd.wait()  # wait until recording is finished
#     sf.write(file_path, audio_data, samplerate)
#     print(f"Saved recording to {file_path}")
#     return file_path


#spk = AudioEngine(16000)
#a = listen_for_voice(spk)

#record_audio("inp.wav")



















# import torch
# import numpy as np
# import sounddevice as sd

# from API.modules.AudioEngine import AudioEngine
# # Load via Torch Hub
# vad_model, _ = torch.hub.load(
#     repo_or_dir="models/snakers4_silero-vad_master",
#     model="silero_vad",
#     source="local"
# )
# vad = vad_model


# SR = 16000
# PastBuffer_Size = 500 #16 sec 
# BUFFER_SIZE = 8000  # 0.5sec
# CHUNK = 512        # ~0.032 sec
# SILENCE_FRAMES = 50 #~1.6sec
# MAX_AUDIO_LENGTH = 15 #15sec

# def listen_for_voice(speaker: AudioEngine):
#     print("Waiting for voice...")
#     once = False
#     with sd.InputStream(samplerate=SR, channels=1, blocksize=CHUNK) as stream:
#         #to check if flow reaching this or not..
#         if not once:
#             print("started getting buffer...")
#             speaker.play_tts_file("effects/mic.mp3",volume=1)
#             once = True
        
#         #variables
#         buffer = []
#         PastBuffer = []
#         started = False
#         recording = []
#         silence_count = 0

#         #main loop
#         while True:
#             #single itration = 1 chunk

#             if silence_count > SILENCE_FRAMES:
#                 print("recording ended")
#                 break

#             audio, _ = stream.read(CHUNK)
#             audio = np.squeeze(audio).astype(np.float32)

#             if started:
#                 recording.extend(audio)
#                 if len(recording) > SR*MAX_AUDIO_LENGTH:
#                     print("[warning] crossing max audio context length!!")
#                     #we will fix this while last polishing!!

#             #will store pre-roll
#             if not started:
#                 buffer.extend(audio)
#                 if len(buffer) > BUFFER_SIZE:
#                     buffer = buffer[-BUFFER_SIZE:]


#             # threshold / fast pre-check + background energy

#             #storing bg energy
#             level = np.abs(audio).mean()
#             PastBuffer.append(level)
#             if len(PastBuffer) > PastBuffer_Size:
#                     PastBuffer = PastBuffer[-PastBuffer_Size:]
            
#             #dynamic threshold
#             meanenv = max(sum(PastBuffer)/len(PastBuffer),1e-3) if PastBuffer else 1e-3
#             vol = ((level/meanenv)*100) -100
#             #delta(in silence) if meanenv < (noise amount) else delta(in noise)
#             before_start = 190 if meanenv < 0.05 else 60 # delta is more if silence and less if too noise
#             after_start = 40
#             if not (vol > (before_start if not started else after_start)): #40-----
#                 #nothing found as activity
#                 if started: #increase silent count when recording and no activity detected
#                     silence_count+=1
#                 continue
#             print(vol,meanenv,vol*meanenv)
#             #if we reach here means we detected activity!!
#             #runing VAD
#             audio_tensor = torch.from_numpy(audio).unsqueeze(0)
#             prob = vad(audio_tensor, 16000).item()
#             print(prob,"probability of speech")
#             if (prob > 0.7):
#                  #finally detect voice

#                 if (not started):
#                     #started recording
#                     started = True
#                     recording.extend(buffer.copy())   # add pre-roll
#                     print("Voice detected, started recording...")
#                     continue
#                 else:
#                     #already recording so we reset silent count
#                     silence_count = 0
#             else:
#                 #if low probability
#                 #then also increase silent count
#                 if started:
#                     silence_count+=1

#     return np.array(recording, dtype=np.float32)
                
            