#!/usr/bin/env python3

import numpy as np
import cv2
import face_recognition
import sys

from multiprocessing import Queue
from multiprocessing.managers import SyncManager
from queue import Queue as ImageQueue
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import setGlobalLogger

setGlobalLogger(None)
print("OpenGL Pipeline")
from pylibfreenect2 import OpenGLPacketPipeline

print("Starting Tracking")

def __draw_bbox(valid, frame, bbox, color, text):
    if not valid:
        return
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2, 1)
    cv2.putText(frame, text, (bbox[0], bbox[1] - 4), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
def __scale_frame(frame, scale_factor = 1):
    if scale_factor == 1:
        return frame
    return cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)

def face_locations(image):
    pass

class NeuronManager(SyncManager):
    pass

NeuronManager.register('get_web_neuron')
NeuronManager.register('get_alexa_neuron')

manager = NeuronManager(address=('', 4081), authkey=b'daisy')
manager.connect()
web_neuron = manager.get_web_neuron()
alexa_neuron = manager.get_alexa_neuron()

faces = {
    "JessePai": "../faces/JPai-1.jpg",
#    "VladMok": "./faces/Vlad.jpg",
#    "TeddyMen": "./faces/TMen-1.jpg"
}

known_faces = {}

for person in faces:
    image = face_recognition.load_image_file(faces[person])
    print(person)
    face_encoding_list = face_recognition.face_encodings(image)
    if len(face_encoding_list) > 0:
        known_faces[person] = face_encoding_list[0]
    else:
        print("\tCould not find face for person...")

pipeline = OpenGLPacketPipeline()

target = "JessePai"

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
face_process_frame = True

bbox = None
track_bbox = None

while True:
    timer = cv2.getTickCount()

    frames = listener.waitForNewFrame()

    color = frames["color"]
    depth = frames["depth"]

    registration.apply(color, depth, undistorted, registered, bigdepth=bigdepth)

    bd = np.resize(bigdepth.asarray(np.float32), (1080, 1920))
    c = cv2.cvtColor(color.asarray(), cv2.COLOR_RGB2BGR)

    face_bbox = None
    new_track_bbox = None
    face_locations = face_recognition.face_locations(c, number_of_times_to_upsample=0, model="cnn")
    face_encodings = face_recognition.face_encodings(c, face_locations)
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
                [known_faces[target]], face_encoding, 0.6)
        if len(matches) > 0 and matches[0]:
            (top, right, bottom, left) = face_locations[0]

            face_bbox = (left, top, right, bottom)
            mid_w = int((left + right) / 2)
            mid_h = int((top + bottom) / 2)

            break
    __draw_bbox(face_bbox is not None, c, face_bbox, (0, 0, 255), target)

    c = __scale_frame(c, scale_factor = 0.5)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    cv2.putText(c, "FPS : " + str(int(fps)), (100,50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
    image = cv2.imencode('.jpg', c)[1].tostring()
    web_neuron.update([('image', image)])

    listener.release(frames)

self.so.close()
cv2.destroyAllWindows()
device.stop()
device.close()


