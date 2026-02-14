from typing import Generator
import requests
import json

class OllamaAPI:
    def __init__(self, model="gemma3:1b", timeout=30):
        self.model = model
        self.url = "http://localhost:11434/api/chat"
        self.timeout = timeout

    def ask(self, messages) -> str:
        if not messages or not isinstance(messages, list):
            return "I don't have enough context to respond."

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }

        try:
            res = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError:
            return "The assistant is not ready yet. Please wait."
        except requests.exceptions.Timeout:
            return "Thinking took too long. Please try again."
        except requests.exceptions.RequestException:
            return "I ran into a system problem. Try again."

        if res.status_code != 200:
            return "I had trouble thinking just now."

        try:
            data = res.json()
        except json.JSONDecodeError:
            return "I couldn't understand my own response."

        # 🔴 THIS is the part you were asking about
        if isinstance(data, dict) and "error" in data:
            # Example: {"error": "out of memory"}
            err = data.get("error", "").lower()

            if "pull" in err or "download" in err:
                return "I am downloading my model. Please wait a bit."

            if "model" in err and "not found" in err:
                return "My language model is not installed."

            if "memory" in err:
                return "I ran out of memory. Please close other apps."
            
            return "Something went wrong while thinking."

        # Normal success path
        message = data.get("message")
        if not isinstance(message, dict):
            return "I got an unexpected response."

        content = message.get("content")
        if not content:
            return "I couldn't form a proper reply."

        return content.strip()
    
    def ask_stream(self, messages):
        if not isinstance(messages, list) or not messages:
            yield "[error] No input provided"
            return

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }

        try:
            res = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout,
                stream=True
            )
        except requests.exceptions.ConnectionError:
            yield "[error] Ollama server is not running"
            return
        except requests.exceptions.Timeout:
            yield "[error] Response timed out"
            return
        except requests.exceptions.RequestException:
            yield "[error] Request failed"
            return

        if res.status_code != 200:
            yield "[error] Ollama returned bad status"
            return

        full_text = ""

        try:
            for line in res.iter_lines(decode_unicode=True):
                if not line:
                    continue

                data = json.loads(line)

                # 🔴 Handle server-side error object
                if "error" in data:
                    err = data["error"].lower()

                    if "model" in err and "not found" in err:
                        yield "[error] Model not installed"
                    elif "memory" in err:
                        yield "[error] Out of memory"
                    else:
                        yield "[error] " + data["error"]
                    return

                # Normal streaming chunk
                if "message" in data:
                    chunk = data["message"].get("content", "")
                    if chunk:
                        full_text += chunk
                        yield chunk

                if data.get("done"):
                    break

        except json.JSONDecodeError:
            yield "[error] Stream corrupted"
            return



'''
def is_model_available(self):
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=5)
        if res.status_code != 200:
            return False

        data = res.json()
        models = [m["name"] for m in data.get("models", [])]
        return self.model in models

    except Exception:
        return False


'''
