#!/usr/bin/env python3

import face_recognition
import cv2
from daisy_spine import DaisySpine
from daisy_eye import DaisyEye
from multiprocessing import Process, Queue
import time

faces = {
    "JessePai": "./faces/JPai-1.jpg",
#    "VladMok": "./faces/VMok-1.jpg",
#    "TeddyMen": "./faces/TMen-1.jpg"
}

name = "JessePai"

data = None

def begin_tracking(name, data_queue):
    print("Begin Tracking")
    eye = DaisyEye(faces, data_queue)
    eye.find_and_track_correcting(name, debug=False)

def daisy_action(data_queue):
    print("Getting Data")
    while True:
        print(data_queue.get())
        time.sleep(5)

if __name__ == "__main__":
    #spine = DaisySpine()
    data = Queue()
    #eye.find_and_track_correcting(name)
    action_p = Process(target = daisy_action, args=(data, ))
    action_p.daemon = True
    action_p.start()
    begin_tracking("JessePai", data)
