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

"""
TEST ORB SIFT(TOO SLOW) and SURF(9-10?)
"""

def different_tracker(cam_num = 1):
    cam = cv2.VideoCapture(cam_num)

    ret, init_frame = camera_prep(cam)

    bbox = select_ROI(init_frame)

    orb = cv2.ORB_create()

    init_frame = init_frame[bbox[1]:bbox[3],bbox[0]:bbox[2],:].copy()


    while True:
        ret, frame = cam.read()

        init_gray = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        init_kpt, init_desc = orb.detectAndCompute(init_gray, None)
        kpt, desc = orb.detectAndCompute(gray, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(init_desc, desc,k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        matchesMask = None
        if len(good) > 5:
            src_pts  = np.float32([ init_kpt[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts  = np.float32([ kpt[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = init_frame.shape[:2]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            print("Not enough matches...")
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

        output_frame = frame[:,:,:].copy()
        cv2.drawMatches(init_frame, init_kpt, frame, kpt, good, output_frame, **draw_params)

        cv2.imshow("Frame", output_frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    different_tracker()

"""
OLD BRAIN MAIN
Dropping the scale factor from 1 means that the face needs to be closer to
the camera. Please make sure the scale factor is greater than 0 and less than
or equal to 1.
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
"""
