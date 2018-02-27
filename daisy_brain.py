#!/usr/bin/env python3

import face_recognition
import cv2
import daisy_test

faces = {
    "JessePai": "./faces/JPai-1.jpg",
    "JessePai": "./faces/JPai-2.jpg",
    "JessePai": "./faces/JPai-3.jpg"
}

tracker_types = ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"]
tracker_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)]

if __name__ == "__main__":
    """
    Dropping the scale factor from 1 means that the face needs to be closer to
    the camera. Please make sure the scale factor is greater than 0 and less than
    or equal to 1.
    """
    #daisy_test.identify_person(faces, "JessePai", scale_factor = 1)
    daisy_test.track_object_all_types(types = ["CSRT"])
    #daisy_test.track_and_id_face(faces, "JessePai")
