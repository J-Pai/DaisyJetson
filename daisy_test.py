import cv2
import dlib
import face_recognition
import time
import numpy as np
from matplotlib import pyplot as plt

def scale_frame(frame, scale = 1):
    if (scale == 1):
        return frame
    return cv2.resize(frame, (0,0), fx=scale, fy=scale)
    # return small_frame[:, :, ::-1]

def identify_faces(cam_num = 1, scale_factor = 1, width = 1024, height = 576):
    video_capture = cv2.VideoCapture(cam_num)

    if not video_capture.isOpened():
        print("Could not open video")
        return

    face_locations = []
    process_this_frame = True

    time.sleep(5)

    video_capture.set(3, width)
    video_capture.set(4, height)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_small_frame = scale_frame(frame, scale_factor)
        timer = cv2.getTickCount()

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left) in face_locations:
            top *= int(1/scale_factor)
            right *= int(1/scale_factor)
            bottom *= int(1/scale_factor)
            left *= int(1/scale_factor)

            print((top, right, bottom, left))
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)


        cv2.putText(frame, "FPS: " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def identify_person(faces, cam_num = 1, scale_factor = 1, width = 1024, height = 576):
    video_capture = cv2.VideoCapture(cam_num)

    if not video_capture.isOpened():
        print("Could not open video")
        return

    known_face_encodings = []
    known_face_names = []

    for person in faces:
        image = face_recognition.load_image_file(faces[person])
        print(person)
        face_encoding_list = face_recognition.face_encodings(image)
        if (len(face_encoding_list) > 0):
            face_encoding = face_encoding_list[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(person)
        else:
            print("\tCould not find face for person...")

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    video_capture.set(3, width)
    video_capture.set(4, height)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_small_frame = scale_frame(frame, scale_factor)
        timer = cv2.getTickCount()

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.6)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                face_names.append(name)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= int(1/scale_factor)
            right *= int(1/scale_factor)
            bottom *= int(1/scale_factor)
            left *= int(1/scale_factor)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

            cv2.rectangle(frame, (left, bottom + 20), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 3, bottom + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


        cv2.putText(frame, "FPS: " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def init_tracker(frame, bbox, tracker_type = "BOOSTING"):
    tracker = None;
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

def camera_prep(video_capture):
    ret, frame = video_capture.read()
    if not ret:
        print("Cannot read video file")
        sys.exit()
    print("Press q when image is ready")
    while True:
        ret, frame = video_capture.read()
        cv2.imshow("Image Prep", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return ret, frame

def select_ROI(frame):
    bbox = cv2.selectROI(frame, False)
    cv2.destroyAllWindows()
    return bbox;

def draw_bbox(valid, frame, bbox, color, text):
    if not valid:
        return
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, color, 2, 1)
    cv2.putText(frame, text, (int(bbox[0]) + 3, int(bbox[1] + bbox[3]) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def track_object_all_types(cam_num = 1, \
        types = ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT", "DLIB"], \
        colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), \
            (0, 255, 255), (255, 0, 255), (255, 255, 255), (50, 0, 0)]):
    video_capture = cv2.VideoCapture(cam_num)

    ret, frame = camera_prep(video_capture)

    bbox = select_ROI(frame)

    trackers = {}

    for tracker in types:
        trackers[tracker] = init_tracker(frame, bbox, tracker)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        timer = cv2.getTickCount()

        tracker_ret_and_bbox = {}
        for tracker in types:
            if tracker != "DLIB":
                tracker_ret_and_bbox[tracker] = trackers[tracker].update(frame)
            elif tracker == "DLIB":
                trackers[tracker].update(frame)
                rect = trackers[tracker].get_position()

                tracker_ret_and_bbox[tracker] = (True,
                        (rect.left(), rect.top(), rect.width(), rect.height()))

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        for tracker in types:
            draw_bbox(tracker_ret_and_bbox[tracker][0], \
                    frame, \
                    tracker_ret_and_bbox[tracker][1], \
                    colors[types.index(tracker)], \
                    tracker)

        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1);

        failedTrackers = "FAILED: "
        for tracker in types:
            if not tracker_ret_and_bbox[tracker][0]:
                failedTrackers += tracker + " "

        cv2.putText(frame, failedTrackers, (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,142), 1)

        cv2.imshow("Tracking", frame)

        if cv2.waitkey(1) & 0xff == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_object_all_types(types = ["CSRT"])
