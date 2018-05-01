import rpyc
from rpyc.utils.server import ThreadedServer

alexa_command = ""

class DaisyNeuron(rpyc.Service):
    def __init__(self, name):
        print("Created Service", name)
    def exposed_get_alexa_command(self):
        return alexa_command
    def exposed_set_alexa_command(self, cmd):
        alexa_command = cmd
        return alexa_command
    def exposed_clear_alexa_command(self):
        alexa_command = ""
        return "Cleared"

if __name__ == "__main__":
    server = ThreadedServer(DaisyNeuron, port=4081)
    server.start()
