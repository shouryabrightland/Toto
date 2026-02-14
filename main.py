from API.serve import serve,Request,Response
from IntentManager.expectation import boost_by_expectation
from IntentManager.intent import detect_intents
#terms
SHUTDOWN = "shutdown"
PVE = "passive"
CLEARM = "clearMemory"


def server(req:Request, res:Response):
    text = req.message
    intents = boost_by_expectation(req.intent, res.current_expectation)
    print(intents)
    
    #shutdown
    if intents.get(SHUTDOWN,0) > 0.4:
        res.send("Do you want me to shut down?")
        res.expecting("YES_NO")
        res.payload[SHUTDOWN] = True
        return
    if res.payload.get(SHUTDOWN,False):
        if (intents.get(SHUTDOWN,0) > 0.75) or intents.get(PVE,0)>0.5:
            res.exit("voice command SHUTDOWN")
            return
        else:
            res.payload[SHUTDOWN] = False
            res.send("okey!")
    
    #clear memory
    if intents.get(CLEARM,0) > 0.5:
        res.send("Do you want me to clear memory?")
        res.expecting("YES_NO")
        res.payload[CLEARM] = True
        return
    if res.payload.get(CLEARM,False):
        if intents.get(PVE,0)>0.5:
            res.chat.clear()
            res.send("cleared chat voice command")
            res.askAI()
            return
        else:
            res.payload[CLEARM] = False
            res.send("okey!")

    #common work...
    #if top["intent"] == SHUTDOWN and top["score"] > 0.4:

    # fallback
    res.askAI(text)


    

serve(server)





