#!/usr/bin/env python3

import face_recognition
import cv2
import daisy_test

faces = {
    "JessePai": "./faces/JPai-1.jpg",
    "JessePai": "./faces/JPai-2.jpg",
    "JessePai": "./faces/JPai-3.jpg"
}

tracker_types = ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE"]

if __name__ == "__main__":
    daisy_test.identify_person(faces, "JessePai", scale_factor = 1)
    #daisy_test.track_object(tracker_types[6])
    #daisy_test.track_object_all_types()
    #daisy_test.track_and_id_face(faces, "JessePai")
