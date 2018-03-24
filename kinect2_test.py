#!/usr/bin/env python3

import numpy as np
import cv2
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

def view_kinect(min_range = 0, max_range = 256):
    fn = Freenect2()
    num_devices = fn.enumerateDevices()

    if num_devices == 0:
        print("No device connected!")

    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial, pipeline = pipeline)

    listener = SyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)

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

        convertedColor = cv2.cvtColor(color.asarray(), cv2.COLOR_RGB2BGR)
        convertedRegis = cv2.cvtColor(registered.asarray(np.uint8), cv2.COLOR_RGB2BGR)
        convertedBD = bigdepth.asarray(np.float32) / 4500 # Max range is 4500, divide by 4500 to get more interesting results...

        cv2.imshow("depth", depth.asarray() / 4500)
        cv2.imshow("color", cv2.resize(convertedColor, (int(1920 / 3), int(1080 / 3))))
        cv2.imshow("registered", convertedRegis)
        cv2.imshow("bigdepth", cv2.resize(convertedBD, (int(1920/ 3), int(1082 / 3))))

        listener.release(frames)

        key = cv2.waitKey(delay=1)
        if key == ord('q'):
            break
    device.stop()
    device.close()

if __name__ == "__main__":
    view_kinect()

