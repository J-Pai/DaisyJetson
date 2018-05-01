import zerorpc

class DaisyNeuron(object):
    state = ""
    def __init__(self):
        print("Setting Up RPC Server")
        self.state = "idle"
    def init_connection(self):
        print("New Connection")
        return "connected"
    def set_state(self, newState):
        self.state = newState
    def get_state(self):
        return self.state

s = zerorpc.Server(DaisyNeuron())
s.bind("tcp://0.0.0.0:4081")
s.run()
