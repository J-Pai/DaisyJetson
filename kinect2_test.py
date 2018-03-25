#!/usr/bin/env python3

import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import setGlobalLogger

setGlobalLogger(None)

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

DEFAULT_FACE_TARGET_BOX = (int(RGB_W/2) - 75, int(RGB_H/2) - 100,
        int(RGB_W/2) + 75, int(RGB_H/2) + 100)
DEFAULT_TRACK_TARGET_BOX = (int(RGB_W/2) - 340, int(RGB_H/2) - 220,
        int(RGB_W/2) + 340, int(RGB_H/2) + 220)

def __draw_bbox(valid, frame, bbox, color, text):
    if not valid:
        return
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2, 1)
    cv2.putText(frame, text, (bbox[0], bbox[1] - 4), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def __init_tracker(frame, bbox, tracker_type = "BOOSTING"):
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

    ret = tracker.init(frame, bbox)

    if not ret:
        return None
    return tracker

def __select_ROI(frame):
    bbox = cv2.selectROI(frame, False)
    cv2.destroyAllWindows()
    return bbox;

def clean_color(color, bigdepth, min_range, max_range):
    #__draw_bbox(True, color, depth_view, (0, 255, 0), "depth_view")a
    color[(bigdepth < min_range) | (bigdepth > max_range)] = 0

def view_kinect(tracker = "CSRT", min_range = 0, max_range = 1000,
        track_target_box = DEFAULT_TRACK_TARGET_BOX,
        face_target_box = DEFAULT_FACE_TARGET_BOX,
        res = (RGB_W, RGB_H),
        video_out = True, debug = True):
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

    initialized = False

    trackerObj = None
    face_count = 5
    face_process_frame = True

    bbox = None
    face_bbox = None
    track_bbox = None

    while True:
        timer = cv2.getTickCount()
        frames = listener.waitForNewFrame()

        color = frames["color"]
        depth = frames["depth"]

        registration.apply(color, depth, undistorted, registered, bigdepth=bigdepth)

        bd = np.resize(bigdepth.asarray(np.float32), (1080, 1920))
        c = cv2.cvtColor(color.asarray(), cv2.COLOR_RGB2BGR)

        # clean_color(c, bd, min_range, max_range)

        if not initialized:
            bbox = __select_ROI(c)
            trackerObj = __init_tracker(c, bbox, tracker)
            initialized = True

        status = False

        if trackerObj is not None:
            status, trackerBBox = trackerObj.update(c)
            bbox = (int(trackerBBox[0]),
                    int(trackerBBox[1]),
                    int(trackerBBox[0] + trackerBBox[2]),
                    int(trackerBBox[1] + trackerBBox[3]))

        if bbox is not None:
            track_bbox = bbox

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if video_out:
            output_frame = c.copy()
            __draw_bbox(status, output_frame, track_bbox, (0, 255, 0), tracker)
            cv2.putText(output_frame, "FPS : " + str(int(fps)), (100,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
            if not status:
                failedTrackers = "FAILED: "
                failedTrackers += tracker + " "
                cv2.putText(output_frame, failedTrackers, (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,142), 1)

            cv2.imshow("color", cv2.resize(output_frame, (int(1920 / 2), int(1080 / 2))))

        listener.release(frames)

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    device.stop()
    device.close()

if __name__ == "__main__":
    view_kinect()

