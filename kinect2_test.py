#!/usr/bin/env python3

import numpy as np
import cv2
import dlib
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

try:
    print("OpenGL Pipeline")
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        print("OpenCL Pipeline")
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        print("CPU Pipeline")
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()

RGB_W = 1920
RGB_H = 1080

DEPTH_W = 1512
DEPTH_H = 1080

depth_view = (int(RGB_W/2) - int(DEPTH_W/2), int(RGB_H/2) - int(DEPTH_H/2),
        int(RGB_W/2) + int(DEPTH_W/2), int(RGB_H/2) + int(DEPTH_H/2))

def __draw_bbox(valid, frame, bbox, color, text):
    if not valid:
        return
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2, 1)
    cv2.putText(frame, text, (bbox[0], bbox[1] - 4), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def __init_tracker(self, frame, bbox, tracker_type = "BOOSTING"):
    tracker = None;

    print("Init Tracker with:", bbox, tracker_type)

    if tracker_type == "BOOSTING":
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == "MIL":
        tracker = cv2.TrackerMIL_create()
    if tracker_type == "KCF":
        tracker = cv2.TrackerKCF_create()
    if tracker_type == "TLD":
        tracker = cv2.TrackerTLD_create()
    if tracker_type == "MEDIANFLOW":
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == "GOTURN":
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == "MOSSE":
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    if tracker_type == "DLIB":
        tracker = dlib.correlation_tracker()
        tracker.start_track(frame, \
                dlib.rectangle(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
        return tracker

    ret = tracker.init(frame, bbox)

    if not ret:
        return None
    return tracker

def __select_ROI(self, frame):
    bbox = cv2.selectROI(frame, False)
    cv2.destroyAllWindows()
    return bbox;

def clean_color(color, bigdepth, min_range, max_range):
    #__draw_bbox(True, color, depth_view, (0, 255, 0), "depth_view")a
    color[(bigdepth < min_range) | (bigdepth > max_range)] = 0

def view_kinect(tracker = "CSRT", min_range = 1000, max_range = 2000):
    fn = Freenect2()
    num_devices = fn.enumerateDevices()

    if num_devices == 0:
        print("No device connected!")

    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial, pipeline = pipeline)

    listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)

    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)

    device.start()

    registration = Registration(device.getIrCameraParams(),
            device.getColorCameraParams())

    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)
    bigdepth = Frame(1920, 1082, 4)

    while True:
        frames = listener.waitForNewFrame()

        color = frames["color"]
        depth = frames["depth"]

        registration.apply(color, depth, undistorted, registered, bigdepth=bigdepth)

        convertedBigDepth = np.resize(bigdepth.asarray(np.float32), (1080, 1920))
        convertedColor = cv2.cvtColor(color.asarray(), cv2.COLOR_RGB2BGR)
        convertedRegis = cv2.cvtColor(registered.asarray(np.uint8), cv2.COLOR_RGB2BGR)

        clean_color(convertedColor, convertedBigDepth, min_range, max_range)

        cv2.imshow("color", cv2.resize(convertedColor, (int(1920 / 3), int(1080 / 3))))
        #cv2.imshow("bigdepth", cv2.resize(convertedBigDepth / 4500, (int(1920/ 3), int(1080 / 3))))

        listener.release(frames)

        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            break
    device.stop()
    device.close()

if __name__ == "__main__":
    view_kinect()

