class Log:
    def __init__(self, name):
        self.name = name or "Logging"
    def log(self,*lis):
        print("["+self.name+"]",*lis)