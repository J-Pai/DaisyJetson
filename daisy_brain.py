#!/usr/bin/env python3

import face_recognition
import cv2
import daisy_test

faces = {
    "JessePai": "./faces/JPai-1.jpg"
}
scale_factor = 1

if __name__ == "__main__":
    #daisy_test.identify_person(faces, "JessePai")
    daisy_test.track_object(1, 1)
