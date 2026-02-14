from IntentManager.expectationData import EXPECTATION_BOOSTS, EXPECTATION_PATTERNS
import re


def boost_by_expectation(intents: dict, expectation: str) -> dict:
    rules = EXPECTATION_BOOSTS.get(expectation)
    if not rules:
        return intents

    for key in intents.keys():
        score = intents[key]

        if score <= 0:
            continue

        boost = rules.get(key)
        if boost:
            intents[key] = min(score + boost, 1.0)
            print("boost",score,intents[key],key)

    return intents


def detect_expectation(last_agent_message: str) -> str:
    if not last_agent_message:
        return "NONE"

    text = last_agent_message.lower()

    for expectation, patterns in EXPECTATION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text):
                print(expectation)
                return expectation
    
    return "NONE"
