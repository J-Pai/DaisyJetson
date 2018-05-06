#!/usr/bin/env python3
import sys
import os
import face_recognition
import cv2
from daisy_spine import DaisySpine
from daisy_spine import Dir
from daisy_eye import DaisyEye
from multiprocessing import Process, Queue
from multiprocessing.managers import SyncManager
import time
import argparse

class NeuronManager(SyncManager):
    pass
NeuronManager.register('get_alexa_neuron')
connected = True
alexa_neuron = None
manager = NeuronManager(address=('', 4081), authkey=b'daisy')
try:
    manager.connect()
    alexa_neuron = manager.get_alexa_neuron()
    print("Brain connected to neuron manager.")
except ConnectionRefusedError:
    print("Brain not connected to neuron manager.")
    connected = False

faces = {
    "Jessie": "./faces/JPai-2.jpg",
#    "Vladimir": "./faces/Vlad.jpg",
    "teddy": "./faces/Teddy-1.jpg"
}

name = "JessePai"
data = None
eye = None

X_THRES = 100
Z_CENTER = 1500
Z_THRES = 100
pid = -1

def begin_tracking(name, data_queue, video=True, stream=True):
    print("Begin Tracking")
    print("Video: ", video)
    eye = DaisyEye(faces, data_queue)
    eye.find_and_track_kinect(None, "CSRT", video_out=video, stream_out=stream)
    data_queue.close()

def daisy_action(data_queue, debug=True):
    spine = DaisySpine()
    print("Getting Data")
    print("Debug: ", debug)
    print(spine.read_all_lines())
    data = None
    prev_statement = ""

    while True:
        state = None
        direction = None
        if connected:
            currNeuron = alexa_neuron.copy()
            if "state" in currNeuron:
                state = currNeuron.get("state")
            if state == "moving":
                direction = currNeuron.get("direction")
        if state is None or state == "idle" or state == "moving":
            if direction is not None:
                out = None
                if direction == "left" or direction == "counterclockwise":
                    out = spine.turn(Dir.CCW)
                elif direction == "right" or direction == "clockwise":
                    out = spine.turn(Dir.CW)
                elif direction == "forward":
                    out = spine.forward()
                elif direction == "backward":
                    out = spine.backward()
                else:
                    out = spine.halt()
                if debug:
                    statement = ("Moving:", direction, out)
            elif debug:
                statement = "Idling"

            if debug and statement != prev_statement:
                prev_statement = statement
                print(statement)
            continue
        if not data_queue.empty():
            data = data_queue.get()
        if data:
            (status, bbox, center, distance, res) = data
            if not status:
                continue
            if status == "STOP":
                break
            center_x = center[0]
            center_y = center[1]

            res_center_x = int(res[0] / 2)
            res_center_y = int(res[1] / 2)

            out = None
            if center_x < res_center_x - X_THRES:
                out = spine.turn(Dir.CW)
            elif center_x > res_center_x + X_THRES:
                out = spine.turn(Dir.CCW)
            elif distance > Z_CENTER + Z_THRES:
                out = spine.forward()
            elif distance < Z_CENTER - Z_THRES:
                out = spine.backward()
            else:
                out = spine.halt()

            if debug:
                statement = (center_x, res_center_x, center, distance, res, out)

            if debug and statement != prev_statement:
                prev_statement = statement
                print(statement)
            data = None
    print("Action Thread Exited")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Daisy's Brain")
    parser.add_argument("--no-debug", action="store_const", const=True, help="Disable debug output")
    parser.add_argument("--no-video", action="store_const", const=True, help="Disable video output")
    parser.add_argument("--no-stream", action="store_const", const=True, help="Disable stream output")
    args = parser.parse_args()
    print("Daisy's Brain is Starting ^_^")
    if connected:
        # Clear alexa neuron.
        alexa_neuron.clear()
    data = Queue()
    action_p = Process(target = daisy_action, args=(data, not args.no_debug, ))
    action_p.daemon = True
    action_p.start()
    pid = action_p.pid
    begin_tracking("JessePai", data, not args.no_video, not args.no_stream)
    action_p.terminate()
    print("Brain Terminated +_+")
