import API.modules.trash.words as wordbook
EXPECTATIONS = {
    "NONE",
    "YES_NO",
    "DETAILS",
    "CONFIRM_EXIT",
    "COMMAND",
} 

def detect_expectation(last_agent_msg: str):
    if not last_agent_msg:
        return "NONE"

    msg = last_agent_msg.lower().strip()

    if msg.endswith("?"):
        if any(w not in msg for w in wordbook.question_starters) or any(w in msg for w in ["yes", "no", "okay", "right", "correct"]):
            return "YES_NO"
        return "DETAILS"

    if any(w in msg for w in ["do you want to exit", "should i stop", "want me to stop"]):
        return "CONFIRM_EXIT"

    if any(w in msg for w in ["say", "tell me", "give me", "run command"]):
        return "COMMAND"

    return "NONE"
