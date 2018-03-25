#!/usr/bin/env python3

import face_recognition
import cv2
from daisy_spine import DaisySpine
from daisy_spine import Dir
from daisy_eye import DaisyEye
from multiprocessing import Process, Queue
import time

faces = {
    "JessePai": "./faces/JPai-1.jpg",
#    "VladMok": "./faces/Vlad.jpg",
#    "TeddyMen": "./faces/TMen-1.jpg"
}

name = "JessePai"
data = None
eye = None

def begin_tracking(name, data_queue):
    print("Begin Tracking")
    eye = DaisyEye(faces, data_queue, flipped = True)
    eye.find_and_track_correcting(name, tracker="CSRT", debug=False)
    data_queue.close()

def daisy_action(data_queue):
    #spine = DaisySpine()
    print("Getting Data")
    #print(spine.read_all_lines())
    data = None
    while True:
        if not data_queue.empty():
            data = data_queue.get()
        if data:
            (string, bbox, distance, res) = data
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)

            res_center_x = int(res[0] / 2)
            res_center_y = int(res[1] / 2)
            # print(center_x, res_center_x, distance)
            if center_x < res_center_x:
                print(spine.turn(Dir.CW))
            elif center_x > res_center_x:
                print(spine.turn(Dir.CCW))
            else:
                print(spine.halt())

if __name__ == "__main__":
    #spine = DaisySpine()
    #eye = DaisyEye(faces)
    #eye.find_and_track_correcting(name)
    data = Queue()
    action_p = Process(target = daisy_action, args=(data, ))
    action_p.daemon = True
    action_p.start()
    begin_tracking("JessePai", data)
