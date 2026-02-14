EXPECTATION_BOOSTS = {
    "CONFIRM_EXIT": {
        "shutdown": 0.25
    },

    "YES_NO": {
        "passive": 0.40,
        "negative": 0.40
    },

    "COMMAND": {
        "command": 0.35
    },

    "DETAILS": {
        "query": 0.30,
        "story": 0.20
    }
}
EXPECTATION_PATTERNS = {
    "YES_NO": [
        r"\b(do you|would you|should i|is it)\b.*\?",
        r"\b(yes or no)\b",
        r"\b(confirm)\b"
    ],

    "CONFIRM_EXIT": [
        r"\b(exit|quit|shutdown|close)\b.*\?",
        r"\bare you sure\b",
        r"\bconfirm exit\b"
    ],

    "DETAILS": [
        r"\b(tell me more)\b",
        r"\b(can you explain|elaborate|expand)\b",
        r"\b(details|more info)\b"
    ],

    "COMMAND": [
        r"\b(say|tell me to|use command)\b",
        r"\b(type|speak)\b.*\bcommand\b"
    ]
}

