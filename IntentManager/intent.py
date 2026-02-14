
import re
from IntentManager.intentData import AGENT_ALIASES, INTENT_CONFIG


def detect_intents(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = set(text.split())
    print(words)

    agent_hit = bool(words & AGENT_ALIASES)
    results = {}

    for intent, cfg in INTENT_CONFIG.items():
        score = 0.0

        # agent relevance
        if cfg["agent_required"]:
            if not agent_hit:
                results[intent] = 0.0
                continue
            score += 0.3

        # keyword relevance
        if cfg["keywords"]:
            hits = words & cfg["keywords"]
            score += (len(hits) / len(cfg["keywords"])) * cfg["base_weight"]

        # clamp
        score = min(score, 1.0)

        results[intent] = round(score, 3)

    
    return results
