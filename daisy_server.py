#!/usr/bin/env python3

from flask import Flask, render_template, Response
from flask_httpauth import HTTPBasicAuth
import argparse
from multiprocessing.managers import SyncManager
from queue import Empty
import io
import base64
import matplotlib.pyplot as plt
from pymongo import MongoClient

MONGODB_URI = "mongodb://Teddy:password@ds253889.mlab.com:53889/records"
client = MongoClient(MONGODB_URI, connectTimeoutMS=30000)
db = client.get_default_database()
memory_records = db.memory_records
exercise_records = db.exercise_records

class NeuronManager(SyncManager):
    pass
NeuronManager.register('get_web_neuron')
NeuronManager.register('get_alexa_neuron')

manager = NeuronManager(address=('', 4081), authkey=b'daisy')
connected = True
try:
    manager.connect()
    print("Eye connected to neuron manager.")
except ConnectionRefusedError:
    print("Eye not connected to neuron manager.")
    self.connected = False


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
    user = None
    if connected:
        alexa_neuron = manager.get_alexa_neuron()
        user = alexa_neuron.get('user')

    return render_template('index.html',
            mem_graph=mem_game_graph(),
            ex_graph=exercise_graph(),
            currUser=user)

def gen():
    while True:
        img = None
        if connected:
            web_neuron = manager.get_web_neuron()
            img = web_neuron.get('image')
        if img is None:
            img = b'\0'
        yield(  b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n' )

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_MEMORY_RECORD(name):
    record = memory_records.find_one({"user":name})
    return record

def get_EXERCISE_RECORD(name):
    record = exercise_records.find_one({"user":name})
    return record

def mem_game_graph():
    if not connected:
        return '<p>Manager is not connected<p>'

    alexa_neuron = manager.get_alexa_neuron()
    record = get_MEMORY_RECORD(alexa_neuron.get('user'))

    if record is None:
        return '<p>No memory game data recorded for user<p>'

    count = record['count'] + 1
    data = record['data']

    xaxis = list(range(1, count))
    yaxis = data
    y_mean = [record['overall_performance']]*len(xaxis)

    fig, ax = plt.subplots()
    data_line = ax.plot(xaxis,yaxis, label='Data', marker='o')
    mean_line = ax.plot(xaxis,y_mean, label='Mean', linestyle='--')

    ax.set(xlabel='Number of times played (#)', ylabel='Percentage Score (%)',
            title='Memory Game Performance Analytics')
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    img = io.BytesIO()

    plt.savefig(img, format='png', bbox_extra_artists=(legend,), bbox_inches='tight')

    plt.close(fig)

    img.seek(0)

    imgData = base64.b64encode(img.getvalue()).decode()

    img.close()

    return '<img class="pure-img" src="data:image/png;base64, {}">'.format(imgData)

def exercise_graph():
    if not connected:
        return 'Manager is not connected'

    alexa_neuron = manager.get_alexa_neuron()
    record = get_EXERCISE_RECORD(alexa_neuron.get('user'))

    if record is None:
        return '<p>No exercise data recorded for user<p>'

    count = record['count'] + 1
    data = record['data']

    xaxis = list(range(1, count))
    yaxis = data
    y_mean = [record['overall_performance']]*len(xaxis)

    fig, ax = plt.subplots()
    data_line = ax.plot(xaxis,yaxis, label='Data', marker='o')
    mean_line = ax.plot(xaxis,y_mean, label='Mean', linestyle='--')

    ax.set(xlabel='Number of times exercised (#)', ylabel='Repetitions (#)',
            title='Exercise Performance Analytics')
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    img = io.BytesIO()

    plt.savefig(img, format='png', bbox_extra_artists=(legend,), bbox_inches='tight')
    img.seek(0)

    plt.close(fig)

    imgData = base64.b64encode(img.getvalue()).decode()

    img.close()

    return '<img class="pure-img" src="data:image/png;base64, {}">'.format(imgData)


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
