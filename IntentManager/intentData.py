# intents.py
INTENT_CONFIG = {
    "shutdown": {
        "keywords": {"shutdown", "shut", "set", "close", "down", "exit", "quit"},
        "agent_required": True,
        "base_weight": 1.0
    },
    "clearMemory":{
        "keywords": {"clear", "memory", "chat", "data", "foget", "forgot", "kill","reset","jet"},
        "agent_required": True,
        "base_weight": 1
    },

    "command": {
        "keywords": {"command", "commend", "comand", "come", "on"},
        "agent_required": True,
        "base_weight": 0.8
    },
    "wake": {
        "keywords": {"weak","up","morning","wake"},
        "agent_required": True,
        "base_weight": 0.4
    },
    "passive": {
        "keywords": {"yes","yeah","ha","yea","yep"},
        "agent_required": False,
        "base_weight": 1.2
    },
    "negative": {
        "keywords": {"no","nope","nahi","na"},
        "agent_required": False,
        "base_weight": 1.1
    },
}

AGENT_ALIASES = {"try", "tri", "tray", "trie", "dry", "drai", "troy","cry","boys","ways","drag","drive"}
