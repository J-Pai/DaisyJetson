import cv2
import face_recognition

def scale_frame(frame, scale = 1):
    if (scale == 1):
        return frame
    return cv2.resize(frame, (0,0), fx=scale, fy=scale)
    # return small_frame[:, :, ::-1]

def identify_person(faces, person, cam_num = 1, scale_factor = 1):
    video_capture = cv2.VideoCapture(cam_num)

    if not video_capture.isOpened():
        print("Could not open video")
        return

    known_face_encodings = []
    known_face_names = []

    for person in faces:
        image = face_recognition.load_image_file(faces[person])
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(person)

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

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
        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= int(1/scale_factor)
            right *= int(1/scale_factor)
            bottom *= int(1/scale_factor)
            left *= int(1/scale_factor)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

            cv2.rectangle(frame, (left, bottom + 20), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
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

def track_object(tracker_type = "BOOSTING", cam_num = 1):

    video_capture = cv2.VideoCapture(cam_num)

    if not video_capture.isOpened():
        print("Could not open video")
        return

    ret, frame = camera_prep(video_capture)

    bbox = select_ROI(frame)

    tracker = init_tracker(frame, bbox, tracker_type)

    if not tracker:
        print("Tracker not recognized")
        sys.exit()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        timer = cv2.getTickCount()
        ret, bbox = tracker.update(frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50),1);
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,170,50), 1);
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def draw_bbox(valid, frame, bbox, color, text):
    if not valid:
        return
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, color, 2, 1)
    cv2.putText(frame, text, (int(bbox[0]) + 3, int(bbox[1] + bbox[3]) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def track_object_all_types(cam_num = 1):
    video_capture = cv2.VideoCapture(cam_num)

    ret, frame = camera_prep(video_capture)

    bbox = select_ROI(frame)

    trackerB = init_tracker(frame, bbox, "BOOSTING")
    trackerMIL = init_tracker(frame, bbox, "MIL")
    trackerKCF = init_tracker(frame, bbox, "KCF")
    trackerTLD = init_tracker(frame, bbox, "TLD")
    trackerMF = init_tracker(frame, bbox, "MEDIANFLOW")
    trackerGOT = init_tracker(frame, bbox, "GOTURN")
    trackerMOS = init_tracker(frame, bbox, "MOSSE")
    trackerCSRT = init_tracker(frame, bbox, "CSRT")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        timer = cv2.getTickCount()

        retB, bboxB = trackerB.update(frame)
        retMIL, bboxMIL = trackerMIL.update(frame)
        retKCF, bboxKCF = trackerKCF.update(frame)
        retTLD, bboxTLD = trackerTLD.update(frame)
        retMF, bboxMF = trackerMF.update(frame)
        retGOT, bboxGOT = trackerGOT.update(frame)
        retMOS, bboxMOS = trackerMOS.update(frame)
        retCSRT, bboxCSRT = trackerCSRT.update(frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        draw_bbox(retB, frame, bboxB, (255, 0, 0), "Boosting")
        draw_bbox(retMIL, frame, bboxMIL, (0, 255, 0), "MIL")
        draw_bbox(retKCF, frame, bboxKCF, (0, 0, 255), "KCF")
        draw_bbox(retTLD, frame, bboxTLD, (0, 0, 0), "TLD")
        draw_bbox(retMF, frame, bboxMF, (255,255,255), "MF")
        draw_bbox(retGOT, frame, bboxGOT, (100,100,100), "GOT")
        draw_bbox(retMOS, frame, bboxMOS, (0,255,255), "MOSSE")
        draw_bbox(retCSRT, frame, bboxCSRT, (140,0,123), "CSRT")

        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1);

        failedTrackers = "FAILED: "
        if not retB:
            failedTrackers += "BOOSTING "
        if not retMIL:
            failedTrackers += "MIL "
        if not retKCF:
            failedTrackers += "KCF "
        if not retTLD:
            failedTrackers += "TLD "
        if not retMF:
            failedTrackers += "MF "
        if not retGOT:
            failedTrackers += "GOT "
        if not retMOS:
            failedTrackers += "MOS "
        if not retCSRT:
            failedTrackers += "CSRT "


        cv2.putText(frame, failedTrackers, (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,175,0), 1)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()



def track_and_id_face(faces, person, tracker_type = "BOOSTING", cam_num = 1, scale_factor = 1):
    print("NOT YET DEVELOPED")



