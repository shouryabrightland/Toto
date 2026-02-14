import re
from API.serve import Response

AGENT_NAMES = r"(try|tri|trivoice|try voice|tri voice)"
COMMAND_WORD = r"(command|commands|cmd)"

COMMAND_REGEX = re.compile(
    rf"\b{AGENT_NAMES}\s+{COMMAND_WORD}\s+(?P<payload>.+)",
    re.IGNORECASE
)

def extract_command(text: str):
    match = COMMAND_REGEX.search(text)
    if not match:
        return None

    payload = match.group("payload").strip()
    return payload

def handle_command(payload: str, res:Response):
    p = payload.lower()

    if p in {"exit", "quit", "shutdown", "stop"}:
        res.exit("user requested shutdown")
        return True

    if p in {"clear memory", "reset memory", "forget chat"}:
        res.chat.clear()
        res.send("Memory cleared.")
        return True

    if p in {"mute", "stop speaking"}:
        #res.speaker.mute()
        return True

    # Unknown command
    res.send("I heard a command, but I do not recognize it.")
    return True
