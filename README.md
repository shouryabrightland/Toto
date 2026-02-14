# 🎙️ TryVoice v1

A local, offline-first voice assistant built in Python.
Designed for low-resource systems like Raspberry Pi.
No cloud. No spying. Full control.

---

## 🚀 Features

* 🎤 **Live voice input** using Whisper (offline)
* 🧠 **Local LLM** via Ollama (no internet required after setup)
* 🔊 **Real-time text-to-speech** using Piper (streaming, no file I/O)
* ⚡ **Low latency streaming** (LLM → TTS chunk by chunk)
* 🧩 **Middleware architecture** (Request / Response like backend servers)
* 🧠 **Soft-coded intent detection** (robust to Whisper mistakes)
* 🧪 **Graceful failure handling**

  * Ollama not running
  * Model downloading
  * Out-of-memory situations
* 🪶 Optimized for **low RAM & CPU**

---

## 🖥️ Requirements

### 1. Python

* Version: **Python 3.9+**
* Download:
  [https://www.python.org/downloads/](https://www.python.org/downloads/)

Check installation:

```bash
python --version
```

---

### 2. Ollama (Local LLM Backend)

* Download:
  [https://ollama.com/download](https://ollama.com/download)

After installing, pull a model:

```bash
ollama pull gemma3:1b
```

Start Ollama server:

```bash
ollama serve
```

⚠️ Ollama **must be running** before starting TryVoice.

---

## 📦 Installation

### Create virtual environment

```bash
python -m venv venv
```

### Activate environment

Linux / macOS:

```bash
source venv/bin/activate
```

Windows (PowerShell):

```bash
venv\Scripts\Activate.ps1
```

---

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Assistant

```bash
python main.py
```

The assistant will:

1. Play startup sound
2. Initialize Whisper, Ollama, TTS
3. Start listening for voice input

---

## 🧠 How It Works (High Level)

```
Microphone
   ↓
Whisper (Speech → Text)
   ↓
Middleware (Intent / Command handling)
   ↓
Ollama (LLM reasoning)
   ↓
Streaming Text
   ↓
Piper TTS (Live Speech Output)
```

No audio files are written during normal operation.

---

## 🗣️ Voice Commands (Examples)

These are handled **without calling the LLM**:

* “Try command shutdown”
* “Try command clear memory”
* “Yes” / “No” confirmations
* Soft exits like “bye”, “leave”, “quit”

The system confirms destructive actions before executing them.

---

## ⚠️ Known Limitations

* Whisper may mis-transcribe words
  Example:
  `try` → `dry`
  `shutdown` → `set down`

This is handled using:

* fuzzy keyword sets
* confidence-based intent scoring
* agent alias matching

---

## 🧪 Troubleshooting

**Ollama not running**

* Assistant will respond with a safe message
* Start Ollama using:

```bash
ollama serve
```

**Model downloading**

* Assistant waits and informs the user

**Out of memory**

* Close heavy apps (browser, VS Code)
* Use smaller model like `gemma3:1b`

---

## 🛠️ Project Status

* ✅ Version 1 complete
* 🔧 Future work:

  * Wake word
  * GPIO / hardware control
  * Better noise handling
  * Multi-agent routing

---

## 📜 License

Open for learning and experimentation.
Use responsibly.
