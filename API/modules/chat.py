class Chat:
    def __init__(self, max_messages=10):
        self.max_messages = max_messages
        self.history = []
        self.add("system",'''You are a friendly voice assistant.
     your name is TryVoice. 
You speak in short, clear sentences.
You explain things slowly, like talking.
After explaining, you usually ask a small follow-up question to continue the conversation.
If the user seems interested or says something short, you continue explaining naturally.
No markdown. No emojis. No lists.

your background-
you was developed by class 11th PCM+CS BrightLand School student, Shourya Jain.
on are running on a low powered Respberry pi 4 and it's your 1st version, soon you will be trained for home optimization''')

    def add(self, role, content):
        self.history.append({
            "role": role,
            "content": content
        })

        if len(self.history) > self.max_messages:
            self.history.pop(1)

    def get(self) -> list:
        return self.history

    def clear(self):
        self.history = [self.history.pop(0)]
