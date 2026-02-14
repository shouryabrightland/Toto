import subprocess
import platform
import signal
import os

class MusicManager:
    def __init__(self):
        self.proc = None
        self.os = platform.system().lower()

    def start(self, file_path: str, loop=True):
        if self.proc:
            return  # already running

        if self.os == "windows":
            # Windows has no native pause/resume, so loop = fire again
            self.proc = subprocess.Popen(
                ["cmd", "/c", "start", "", file_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        else:
            # Linux / Raspberry Pi
            cmd = ["mpg123", "-q"]
            if loop:
                cmd.append("--loop")
                cmd.append("-1")
            cmd.append(file_path)

            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

    def stop(self):
        if not self.proc:
            return

        if self.os == "windows":
            self.proc.terminate()
        else:
            self.proc.terminate()

        self.proc = None

    def pause(self):
        if not self.proc:
            return

        if self.os != "windows":
            self.proc.send_signal(signal.SIGSTOP)

    def resume(self):
        if not self.proc:
            return

        if self.os != "windows":
            self.proc.send_signal(signal.SIGCONT)

    def is_playing(self):
        return self.proc is not None and self.proc.poll() is None
