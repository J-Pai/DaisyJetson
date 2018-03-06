#!/usr/bin/env python3

import face_recognition
import cv2
import daisy_test
import daisy_eye
from daisy_spine import DaisySpine
from daisy_eye import DaisyEye

faces = {
    "JessePai": "./faces/JPai-1.jpg",
#    "VladMok": "./faces/VMok-1.jpg",
#    "TeddyMen": "./faces/TMen-1.jpg"
}

if __name__ == "__main__":
    #spine = DaisySpine()
    """
    Dropping the scale factor from 1 means that the face needs to be closer to
    the camera. Please make sure the scale factor is greater than 0 and less than
    or equal to 1.
    """
    #daisy_test.identify_faces(scale_factor = 1)
    #daisy_test.identify_person(faces, scale_factor = 1)
    #daisy_test.track_object_all_types(types = ["CSRT"])
    #daisy_test.track_object_all_types(types = ["DLIB"])
    #bbox = daisy_test.id_and_track_face(faces, "JessePai")


    eye = DaisyEye(faces)
    #eye.locate_target("JessePai", debug = True, ret = False)
    #eye.find_and_track("JessePai", debug = False)
    #eye.track_object(video_out = True)
    eye.find_and_track_correcting("JessePai", tracker="CSRT", debug = False)
    #eye.view(bbox_list=[(350,250,450,350),(500,250,600,350), \
    #        (350,400,450,650),(500,400,600,650),(450,350,550,450)])
