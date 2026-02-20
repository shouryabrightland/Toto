import random
import time
from API.modules.AudioEngine import AudioEngine
from API.modules.Logging import Log
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

class serve:
    def __init__(self,server: FunctionType):
        self.log = Log("Server").log
        self.log("Loading Services...")
        whisper = WhisperAPI()
        #cli ollama
        ollama = OllamaAPI()
        chat = Chat(max_messages=50)
        self.log("Awaking Agent..")

        Req = Request(whisper,chat=chat,tts=tts,audio=Audio)
        Res = Response(ollama,tts,chat,Audio)
        recorder = Recorder.WakeAssistant("models/toto_v1.onnx",
                                          speaker = Audio,
                                          whisper_api=whisper,
                                          onDetect=Res.end,
                                          onStopRecording=self.thinking,
                                          )
        Res.askAI()
        Userqueue = Queue()
        threading.Thread(target=recorder.start, daemon=True, args=(Userqueue,)).start()
        while not Res.isTerminated:
            Req.message = Userqueue.get()
            Req.start()
            server(Req,Res,log=Log("Server Insider").log)
            Res.end()

    def thinking(self):
        Audio.play_bg_file("effects/think.mp3", volume=0.5)

class Request:
    def __init__(self,whisper: WhisperAPI,chat:Chat,tts:PiperTTS,audio:AudioEngine):
        self.whisper = whisper
        self.message = None
        self.intent = None
        self.chat = chat
        self.tts = tts
        self.audio = audio
        self.log = Log("Server REQ").log


    def start(self):

        #function that ask for request input
        self.audio.stop_bg()
        self.log("waiting for tts to shut it's mouth")
        self.tts.q.join() #wait for tts to complete..
        self.log("waiting for speaker to shut it's mouth")
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
        self.log = Log("Server RES").log
        self.stopflag = threading.Event()

        # self.ACKS = [
        #     "Alright.",
        #     "Okay.",
        #     "Got it.",
        #     "One moment.",
        #     "Working on that."
        # ]
        # self.ACK_MIN_INTERVAL = 2.5 
    
    def askAI(self,message: str = None,fastOut = True):
        self.stopflag.clear() #reseting before starting new
        '''
        it's a high level function for ollama request
        :param flag: thread's flag when triggured then it will stop...
        :param message: message to insert in chat before giving to Ollama
        :type message: str
        :param fastOut: what instant out or smooth out
        :type fastOut: bool
        '''
        #global current_expectation
        if message:
            self.chat.add("user",message)

        buffer = ""
        res = ""

        for chunk in self.ollama.ask_stream(self.chat.get()):
            #time.sleep(random.uniform(0.2, 0.5))

            # ---------- error ----------
            if chunk.startswith("[error]"):
                self.log(chunk)
                self.tts.enqueue(chunk)
                break

            # ---------- normal buffering ----------
            if fastOut:
                buffer += chunk.strip("\n")
                if buffer.endswith((".", "?", "!", ",")):

                    self.speaker.stop_bg()
                    self.tts.enqueue(buffer)
                    self.log("Buffer given to TTS:",buffer)
                    res += buffer
                    buffer = ""
            else:
                buffer += chunk
            
            #---------------Stopflag------------
            if self.stopflag.is_set():
                self.ollama.stop()
                break
        
        else:
            #for last remaining chunks... or final out for slow option
            if buffer.strip():
                self.speaker.stop_bg()
                self.log("Buffer given to TTS:",buffer)
                self.tts.enqueue(buffer)
                res += buffer
                #self.current_expectation = detect_expectation(res)
                buffer = ""
        
        self.chat.add("assistant", res)

    def end(self):
        self.stopflag.set()

    # def exit(self,msg: str = None):
    #     msg = "Shut Down with no message" if not msg else "shutdown due to "+str(msg)
    #     self.tts.enqueue(msg)
    #     print(msg)
    #     self.speaker.stop_bg()
    #     self.speaker.q.join()
    #     #self.tts.writeWAV(msg)
    #     #self.speaker.play_file("output.wav")
        
    #     self.isTerminated = True
