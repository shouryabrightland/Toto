from API.modules.Logging import Log
from Server.serve import serve,Request,Response

def server(req:Request, res:Response,log:Log.log):
    text = req.message
    #res.askAI(text)
    log("this message is from main")
    res.askAI(req.message)
    print(text)
serve(server)





