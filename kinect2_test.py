#!/usr/bin/env python3

import numpy as np
import cv2
import sys
import face_recognition
import dlib
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

DEFAULT_FACE_TARGET_BOX = (int(RGB_W/2) - 125, int(RGB_H/2) - 125,
        int(RGB_W/2) + 125, int(RGB_H/2) + 125)
DEFAULT_TRACK_TARGET_BOX = (int(RGB_W/2) - 340, int(RGB_H/2) - 220,
        int(RGB_W/2) + 340, int(RGB_H/2) + 220)

CORRECTION_THRESHOLD =  0.50

FACE_COUNT = 0

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
def __scale_frame(frame, scale_factor = 1):
    if scale_factor == 1:
        return frame
    return cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)

def __bbox_overlap(bbox1, bbox2):
    if not bbox1 or not bbox2:
        return 0

    left = max(bbox1[0], bbox2[0])
    top = max(bbox1[1], bbox2[1])
    right = min(bbox1[2], bbox2[2])
    bottom = min(bbox1[3], bbox2[3])

    if left < right and top < bottom:
        return __bbox_area((left, top, right, bottom))
    return 0

def __bbox_area(bbox):
    if not bbox:
        return 0
    (left, top, right, bottom) = bbox
    return (right - left) * (bottom - top)

def __crop_frame(frame, crop_box):
    return frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2],:].copy()

def __clean_color(color, bigdepth, min_range, max_range):
    color[(bigdepth < min_range) | (bigdepth > max_range)] = 0

faces = {
    "JessePai": "./faces/JPai-1.jpg",
#    "VladMok": "./faces/Vlad.jpg",
#    "TeddyMen": "./faces/TMen-1.jpg"
}

def find_and_track_kinect(name, tracker = "CSRT",
        min_range = 1000, max_range = 1700,
        face_target_box = DEFAULT_FACE_TARGET_BOX,
        res = (RGB_W, RGB_H),
        video_out = True, debug = True):

    known_faces = {}

    for person in faces:
        image = face_recognition.load_image_file(faces[person])
        print(person)
        face_encoding_list = face_recognition.face_encodings(image)
        if len(face_encoding_list) > 0:
            known_faces[person] = face_encoding_list[0]
        else:
            print("\t Could not find face for person...")

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


    trackerObj = None
    face_count = 5
    face_process_frame = True

    bbox = None
    face_bbox = None
    track_bbox = None

    while True:
        timer = cv2.getTickCount()

        person_found = False

        frames = listener.waitForNewFrame()

        color = frames["color"]
        depth = frames["depth"]

        registration.apply(color, depth, undistorted, registered, bigdepth=bigdepth)

        bd = np.resize(bigdepth.asarray(np.float32), (1080, 1920))
        c = cv2.cvtColor(color.asarray(), cv2.COLOR_RGB2BGR)

        __clean_color(c, bd, min_range, max_range)

        if face_process_frame:
            small_c = __crop_frame(c, face_target_box)
            face_locations = face_recognition.face_locations(small_c, model="cnn")
            face_encodings = face_recognition.face_encodings(small_c, face_locations)
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                        [known_faces[name]], face_encoding, 0.6)
                if len(matches) > 0 and matches[0]:
                    person_found = True
                    face_count += 1
                    (top, right, bottom, left) = face_locations[0]

                    left += face_target_box[0]
                    top += face_target_box[1]
                    right += face_target_box[0]
                    bottom += face_target_box[1]

                    face_bbox = (left, top, right, bottom)

                    person_found = True

                    break
        face_process_frame = not face_process_frame

        overlap_pct = 0
        track_area = __bbox_area(track_bbox)
        if track_area > 0 and face_bbox:
            overlap_area = __bbox_overlap(face_bbox, track_bbox)
            overlap_pct = min(overlap_area / __bbox_area(face_bbox),
                    overlap_area / __bbox_area(track_bbox))

        if person_found and face_count >= FACE_COUNT and overlap_pct < CORRECTION_THRESHOLD:
            bbox = (face_bbox[0],
                    face_bbox[1],
                    face_bbox[2] - face_bbox[0],
                    face_bbox[3] - face_bbox[1])
            trackerObj = __init_tracker(c, bbox, tracker)
            face_count = 0

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

        w = 0
        h = 0

        if status:
            w = track_bbox[0] + int((track_bbox[2] - track_bbox[0])/2)
            h = track_bbox[1] + int((track_bbox[3] - track_bbox[1])/2)
            if (w < res[0] and w >= 0 and h < res[1] and h >= 0):
                distanceAtCenter =  bd[h][w]
                __update_individual_position("NONE", track_bbox, distanceAtCenter, res)

        if video_out:
            cv2.line(c, (w, 0),
                    (w, int(res[1])), (0,255,0), 1)
            cv2.line(c, (0, h), \
                    (int(res[0]), h), (0,255,0), 1)

            __draw_bbox(True, c, face_target_box, (255, 0, 0), "FACE_TARGET")
            __draw_bbox(status, c, track_bbox, (0, 255, 0), tracker)
            __draw_bbox(person_found, c, face_bbox, (0, 0, 255), name)

            c = __scale_frame(c, scale_factor = 0.5)

            cv2.putText(c, "FPS : " + str(int(fps)), (100,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
            if not status:
                failedTrackers = "FAILED: "
                failedTrackers += tracker + " "
                cv2.putText(c, failedTrackers, (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,142), 1)

            cv2.imshow("color", c)

        listener.release(frames)

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    device.stop()
    device.close()

def __update_individual_position(str_pos, track_bbox, distance, res):
    print(str_pos, track_bbox, distance, res)
if __name__ == "__main__":
    find_and_track_kinect("JessePai")

