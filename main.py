from Server.serve import serve,Request,Response

def server(req:Request, res:Response):
    text = req.message
    #res.askAI(text)
    print(text)
serve(server)





