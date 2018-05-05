#!/usr/bin/env python3

from flask import Flask, render_template, Response
from flask_httpauth import HTTPBasicAuth
import argparse
from multiprocessing.managers import SyncManager
from queue import Empty

class NeuronManager(SyncManager):
    pass

NeuronManager.register('get_web_neuron')
manager = NeuronManager(address=('', 4081), authkey=b'daisy')
manager.connect()

app = Flask(__name__)
auth = HTTPBasicAuth()

users = {
    "daisy_login": "iknowthisisinsecure"
}

@auth.get_password
def get_pw(username):
    if username in users:
        return users.get(username)
    return None

@app.route('/')
@auth.login_required
def index():
    return render_template('index.html')

def gen():
    while True:
        web_neuron = manager.get_web_neuron()
        img = b'\0'
        if 'image' in web_neuron.keys():
            img = web_neuron.get('image')
        yield(  b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n' )

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Daisy's Internet Connection")
    parser.add_argument("--set-ip",
            dest="ip",
            default="localhost",
            help="Specify the IP address to use for initialization")
    parser.add_argument("--set-port",
            dest="port",
            default="8080",
            help="Specify the port to use for initialization")
    args = parser.parse_args()
    app.run(args.ip, int(args.port), threaded=True)
