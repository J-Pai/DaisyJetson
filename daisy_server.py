#!/usr/bin/env python3

from flask import Flask, render_template, Response
import argparse
from multiprocessing.managers import BaseManager

class QueueManager(BaseManager):
    pass
QueueManager.register('get_image_queue')
m = QueueManager(address=('', 4081), authkey=b'daisy')
m.connect()
q = m.get_image_queue()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        yield(  b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + q.get() + b'\r\n' )

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
    app.run(args.ip, int(args.port))
