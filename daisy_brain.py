#!/usr/bin/env python3
import sys
import face_recognition
import cv2
from daisy_spine import DaisySpine
from daisy_spine import Dir
from daisy_eye import DaisyEye
from multiprocessing import Process, Queue
import time
import argparse
import zerorpc

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

def begin_tracking(name, data_queue, video=True):
    print("Begin Tracking")
    print("Video: ", video)
    eye = DaisyEye(faces, data_queue)
    eye.find_and_track_kinect(name, "CSRT", video_out=video)
    data_queue.close()

def daisy_action(data_queue, debug=True):
    rpc = zerorpc.Client()
    rpc.connect("tcp://0.0.0.0:4081")
    if rpc.init_connection() != "connected":
        print("Action No RPC")
        data_queue.put("Fail")
        sys.exit()
    spine = DaisySpine()
    print("Getting Data")
    print("Debug: ", debug)
    print(spine.read_all_lines())
    data = None
    while True:
        rpc_state = rpc.get_state()
        if not data_queue.empty():
            data = data_queue.get()
        if "track" in rpc_state and data:
            (status, bbox, center, distance, res) = data
            if not status:
                continue
            if status == "STOP":
                break
            center_x = center[0]
            center_y = center[1]

            res_center_x = int(res[0] / 2)
            res_center_y = int(res[1] / 2)
            if debug:
                print(center_x, res_center_x, center, distance, res)
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
            """
            if debug and out is not None:
                print(out)
            """
            data = None
    print("Action Thread Exited")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Daisy's Brain")
    parser.add_argument("--no-debug", action="store_const", const=True, help="Disable debug output")
    parser.add_argument("--no-video", action="store_const", const=True, help="Disable video output")
    args = parser.parse_args()
    print("Daisy's Brain is Starting ^_^")
    data = Queue()
    action_p = Process(target = daisy_action, args=(data, not args.no_debug, ))
    action_p.daemon = True
    action_p.start()
    pid = action_p.pid
    begin_tracking("JessePai", data, not args.no_video)
    action_p.terminate()
    print("Brain Terminated +_+")
