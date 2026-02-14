import re
import API.modules.trash.words as wordbook

INTENTS = ["exit", "command", "passive", "query", "story","negative"]

AGENT_TERMS = [
    "try", "tri", "trie", "tray",
    "trivoice", "tri voice", "try voice",
    "assistant", "agent"
]

AGENT_EXPECTATIONS = {
    "NONE",
    "YES_NO",
    "CONFIRM_EXIT",
    "DETAILS",
    "COMMAND"
}

INTENT_PRIORITY = [
    "exit",
    "command",
    "query",
    "story",
    "passive",
    "negative"
]

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def agent_reference_score(text: str) -> float:
    score = 0
    words = set(text.split())
    for term in AGENT_TERMS:
        if term in words:
            score += 0.15
    return min(score, 0.4)

def amplify_with_agent_reference(scores: dict, text: str) -> dict:
    boost = agent_reference_score(text)
    if boost == 0:
        return scores
    ALLOWED = {"query", "command", "story"}
    out = {}
    for k, v in scores.items():
        out[k] = min(v + boost, 1.0) if v > 0 and k in ALLOWED else v
    return out

def apply_expectation_boost(scores: dict, expectation: str) -> dict:
    out = scores.copy()

    def boost(intent, value):
        if out.get(intent, 0) > 0:
            out[intent] = min(out[intent] + value, 1.0)

    if expectation == "YES_NO":
        boost("passive", 0.3)
        boost("negative", 0.3)

    elif expectation == "CONFIRM_EXIT":
        boost("exit", 0.4)

    elif expectation == "DETAILS":
        boost("query", 0.3)
        boost("story", 0.2)

    elif expectation == "COMMAND":
        boost("command", 0.3)

    return out

def prune_zero_intents(scores: dict):
    return {k: v for k, v in scores.items() if v > 0}

def resolve_intent(scores: dict):
    if not scores:
        return None, 0.0

    max_score = max(scores.values())
    candidates = [k for k, v in scores.items() if v == max_score]

    for intent in INTENT_PRIORITY:
        if intent in candidates:
            return intent, scores[intent]

    return None, 0.0

def detect_intents(raw_text: str, expectation: str = "NONE"):
    text = normalize(raw_text)
    words = text.split()

    scores = {i: 0.0 for i in INTENTS}

    if not words:
        return {}, None, 0.0

    # EXIT
    exit_tokens = {"bye", "goodbye", "exit", "quit", "leave"}
    bad_phrases = ["go away", "shut up", "fuck off"]

    for w in words:
        if w in exit_tokens:
            scores["exit"] += 0.6

    if text in bad_phrases:
        scores["exit"] = 1.0

    if len(words) > 6:
        scores["exit"] *= 0.4

    # COMMAND
    primary = ["try", "tri", "tie", "tray", "dry"] + AGENT_TERMS
    secondary = {"command", "commands", "commend", "comand"}

    if any(w in primary for w in words) and any(w in secondary for w in words):
        scores["command"] = 1.0

    # PASSIVE
    negative_words = ["nah","no","nahi","nope"] + bad_phrases
    passive_words = {"hmm", "hm", "ok", "okay", "yes", "yeah", "alright"}
    act = 0
    for w in passive_words:
        if w in words:
            scores["passive"] += 0.4
            act+=1
    for w in negative_words:
        if w in words:
            scores["negative"] += 0.4
            act+=1
    if not act:
        scores["passive"] = 0.2
    del act

    # QUERY
    question_starters = wordbook.question_starters

    if "?" in raw_text or (words and words[0] in question_starters):
        scores["query"] = 1.0

    # STORY
    if len(words) >= 6 and scores["query"] == 0:
        scores["story"] = min(1.0, len(words) / 10)

    #print(scores)
    scores = amplify_with_agent_reference(scores, text)
    #print(scores)
    scores = apply_expectation_boost(scores, expectation)
    #print(scores)
    intent, confidence = resolve_intent(scores)
    #scores = prune_zero_intents(scores)

    return scores, intent, confidence















































# import re

# INTENTS = ["exit", "command", "passive", "query", "story"]

# AGENT_TERMS = {
#     "try", "tri", "trie", "tray",
#     "trivoice", "tri voice", "try voice",
#     "assistant", "agent"
# }
# def agent_reference_score(text: str) -> float:
#     text = text.lower()
#     score = 0.0

#     for term in AGENT_TERMS:
#         if term in text:
#             score += 0.15   # small bump per hit

#     return min(score, 0.4)


# def amplify_with_agent_reference(intent_scores: dict, text: str):
#     boost = agent_reference_score(text)

#     if boost == 0:
#         return intent_scores

#     amplified = {}

#     for intent, score in intent_scores.items():
#         if score > 0:
#             amplified[intent] = min(score + boost, 1.0)
#         else:
#             amplified[intent] = 0

#     return amplified


# def normalize(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"[^\w\s]", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text


# def detect_intents(raw_text: str) -> dict:
#     text = normalize(raw_text)
#     words = text.split()

#     scores = {intent: 0.0 for intent in INTENTS}

#     if not words:
#         return scores

#     # -----------------------
#     # EXIT SIGNALS
#     # -----------------------
#     exit_tokens = {"bye", "goodbye", "exit", "quit", "leave"}
#     exit_phrases = {"go away", "shut up", "fuck off"}

#     for w in words:
#         if w in exit_tokens:
#             scores["exit"] += 0.6

#     if text in exit_phrases:
#         scores["exit"] = 1.0

#     # long sentences reduce exit confidence
#     if len(words) > 6:
#         scores["exit"] *= 0.4

#     # -----------------------
#     # COMMAND MODE
#     # -----------------------
#     primary = {"try", "tri", "tie", "tray", "dry"}
#     secondary = {"command", "commands", "commend", "comand"}

#     if any(w in primary for w in words) and any(w in secondary for w in words):
#         scores["command"] = 1.0

#     # -----------------------
#     # PASSIVE
#     # -----------------------
#     passive = {"hmm", "hm", "ok", "okay", "yes", "yeah", "alright"}

#     if len(words) <= 2 and all(w in passive for w in words):
#         scores["passive"] = 0.8

#     # -----------------------
#     # QUERY
#     # -----------------------
#     question_starters = {"why", "how", "what", "when", "where", "who"}

#     if "?" in raw_text or words[0] in question_starters:
#         scores["query"] = 1.0

#     # -----------------------
#     # STORY / STATEMENT
#     # -----------------------
#     if len(words) >= 6 and scores["query"] == 0:
#         scores["story"] = min(1.0, len(words) / 10)

#     scores = amplify_with_agent_reference(scores,text)
#     return prune_zero_intents(scores)
