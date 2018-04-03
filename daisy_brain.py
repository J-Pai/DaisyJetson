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

X_THRES = 100
Z_CENTER = 1500
Z_THRES = 100
pid = -1

def begin_tracking(name, data_queue):
    print("Begin Tracking")
    eye = DaisyEye(faces, data_queue, cam_num = -1, flipped = True)
    eye.find_and_track_kinect(name, "CSRT", debug=False)
    data_queue.close()

def daisy_action(data_queue):
    spine = DaisySpine()
    print("Getting Data")
    print(spine.read_all_lines())
    data = None
    while True:
        if not data_queue.empty():
            data = data_queue.get()
        if data:
            (string, bbox, distance, res) = data
            if string == "STOP":
                break
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)

            res_center_x = int(res[0] / 2)
            res_center_y = int(res[1] / 2)
            print(center_x, res_center_x, distance, res)
            if center_x < res_center_x - X_THRES:
                print(spine.turn(Dir.CW))
            elif center_x > res_center_x + X_THRES:
                print(spine.turn(Dir.CCW))
            elif distance > Z_CENTER + Z_THRES:
                print(spine.forward())
            elif distance < Z_CENTER - Z_THRES:
                print(spine.backward())
            else:
                print(spine.halt())
            data = None
    print("Action Thread Exited")

if __name__ == "__main__":
    #spine = DaisySpine()
    #eye = DaisyEye(faces)
    #eye.find_and_track_correcting(name)
    data = Queue()
    action_p = Process(target = daisy_action, args=(data, ))
    action_p.daemon = True
    action_p.start()
    pid = action_p.pid
    begin_tracking("JessePai", data)
    action_p.terminate()
    print("Brain Terminated")
