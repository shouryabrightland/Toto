import random
import time
from API.modules.AudioEngine import AudioEngine
from API.piper_api import PiperTTS

Audio = AudioEngine(16000)
tts = PiperTTS("voices/en_US-lessac-low.onnx",Audio)
#print("It's TryVoice")
Audio.play_file("effects/try.wav")
Audio.play_bg_file("effects/song.mp3")

from API.whisper_api import WhisperAPI
from API.ollama_api import OllamaAPI
from API.modules.chat import Chat
from types import FunctionType
from queue import Queue
import threading
import Server.Recorder as Recorder

from IntentManager.expectation import detect_expectation
from IntentManager.intent import detect_intents

class serve:
    def __init__(self,server: FunctionType):
        print("Loading Services...")
        whisper = WhisperAPI(speaker=Audio)
        recorder = Recorder.WakeAssistant("models/toto_v1.onnx",whisper_api=whisper)
        #cli ollama
        ollama = OllamaAPI()
        chat = Chat(max_messages=50)

        print("Awaking Agent..")
        Req = Request(whisper,chat=chat,tts=tts,audio=Audio)
        Res = Response(ollama,tts,chat,Audio)
        Res.askAI()
        Userqueue = Queue()
        threading.Thread(target=recorder.start, daemon=True, args=(Userqueue,)).start()
        while not Res.isTerminated:
            Req.message = Userqueue.get()
            server(Req.listen(),Res)

class Request:
    def __init__(self,whisper: WhisperAPI,chat:Chat,tts:PiperTTS,audio:AudioEngine):
        self.whisper = whisper
        self.message = None
        self.intent = None
        self.chat = chat
        self.tts = tts
        self.audio = audio


    def listen(self):

        #function that ask for request input
        self.audio.stop_bg()
        print("waiting for tts to shut it's mouth")
        self.tts.q.join() #wait for tts to complete..
        print("waiting for speaker to shut it's mouth")
        self.audio.q.join() #wait for speaker 
        return self

    


class Response:
    def __init__(self,ollamaAPI:OllamaAPI,tts:PiperTTS,chat:Chat,speaker:AudioEngine):
        self.isTerminated = False
        self.current_expectation = None
        self.ollama = ollamaAPI
        self.tts = tts
        self.chat = chat
        self.speaker = speaker
        self.payload = {}

        self.ACKS = [
            "Alright.",
            "Okay.",
            "Got it.",
            "One moment.",
            "Working on that."
        ]
        self.ACK_MIN_INTERVAL = 2.5 


    def expecting(self,expect: str):
        print("updating expectation" , expect)
        self.current_expectation = expect

    def send(self,message):
        #to clear bg audio
        self.speaker.stop_bg()
        self.tts.enqueue(message)
        self.chat.add("user",message)
        print("[Manual]",message)

    def exit(self,msg: str = None):
        msg = "Shut Down with no message" if not msg else "shutdown due to "+str(msg)
        self.tts.enqueue(msg)
        print(msg)
        self.speaker.stop_bg()
        self.speaker.q.join()
        #self.tts.writeWAV(msg)
        #self.speaker.play_file("output.wav")
        
        self.isTerminated = True
    
    def askAI(self,message: str = None,fastOut = True):
        #global current_expectation
        if message:
            self.chat.add("user",message)

        buffer = ""
        res = ""
        last_ack_time = 0
        ack_used = False

        for chunk in self.ollama.ask_stream(self.chat.get()):
            time.sleep(random.uniform(0.2, 0.5))

            # ---------- error ----------
            if chunk.startswith("[error]"):
                print(chunk)
                self.tts.enqueue(chunk)
                break
            
            print(chunk, end="", flush=True)

            # ---------- ACK injection (latency mask) ----------
            now = time.time()
            if (
                not ack_used
                and now - last_ack_time > self.ACK_MIN_INTERVAL
                and buffer.strip() == ""   # only before speech starts
            ):
                ack = random.choice(self.ACKS)
                self.speaker.stop_bg()
                self.tts.enqueue(ack,slow=True)
                last_ack_time = now
                ack_used = True

            # ---------- normal buffering ----------
            if fastOut:
                buffer += chunk.strip("\n")
                if buffer.endswith((".", "?", "!", ",")):
                    print("using stream")

                    self.speaker.stop_bg()
                    self.tts.enqueue(buffer)
                    buffer = ""
            else:
                buffer += chunk
        else:
            #for last remaining chunks... or final out for slow option
            if buffer.strip():
                self.speaker.stop_bg()
                self.tts.enqueue(buffer)
                res += buffer
                self.current_expectation = detect_expectation(res)
                buffer = ""
        
        self.chat.add("assistant", res)
