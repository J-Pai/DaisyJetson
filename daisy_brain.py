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
eye = None

def begin_tracking(name, data_queue):
    print("Begin Tracking")
    eye = DaisyEye(faces, data_queue)
    eye.find_and_track_correcting(name, tracker="CSRT", debug=False)
    data_queue.close()

def daisy_action(data_queue):
    spine = DaisySpine()
    print("Getting Data")
    while True:
        data = None
        while not data_queue.empty():
            data = data_queue.get()
        if data:
            print(data)
        print(spine.forward())
        time.sleep(1)

if __name__ == "__main__":
    #spine = DaisySpine()
    #eye = DaisyEye(faces)
    #eye.find_and_track_correcting(name)
    data = Queue()
    action_p = Process(target = daisy_action, args=(data, ))
    action_p.daemon = True
    action_p.start()
    begin_tracking("JessePai", data)
