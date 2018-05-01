import zerorpc

class DaisyNeuron(object):
    state = ""
    recveived = False
    def __init__(self):
        print("Setting Up RPC Server")
        self.state = "idle"
        self.received = False
    def set_state(self, newState):
        self.state = newState
        self.received = False
    def get_state(self):
        self.received = True
        return self.state

s = zerorpc.Server(DaisyNeuron())
s.bind("tcp://0.0.0.0:4081")
s.run()
