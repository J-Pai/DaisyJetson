#!/usr/bin/env python3

import face_recognition
import cv2
import daisy_test
import daisy_eye
from daisy_spine import DaisySpine

faces = {
    "JessePai": "./faces/JPai-1.jpg",
#    "VladMok": "./faces/VMok-1.jpg",
#    "TeddyMen": "./faces/TMen-1.jpg"
}

tracker_types = ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"]
tracker_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]

if __name__ == "__main__":
    #spine = DaisySpine()
    """
    Dropping the scale factor from 1 means that the face needs to be closer to
    the camera. Please make sure the scale factor is greater than 0 and less than
    or equal to 1.
    """
    #daisy_test.identify_person(faces, scale_factor = 1)
    daisy_test.track_object_all_types(types = ["CSRT"])
    #bbox = daisy_test.id_and_track_face(faces, "JessePai")
