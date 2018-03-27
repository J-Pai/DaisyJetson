import cv2
import dlib
import face_recognition
import sys
from multiprocessing import Queue

EXPOSURE_1 = 0
EXPOSURE_2 = 0

RGB_W = 800
RGB_H = 600

FACE_W = RGB_W
FACE_H = RGB_H
DEFAULT_FACE_TARGET_BOX = (int(FACE_W/2) - 75, int(FACE_H/2) - 100,
        int(FACE_W/2) + 75, int(FACE_H/2) + 100)

TRACK_W = RGB_W
TRACK_H = RGB_H
DEFAULT_TRACK_TARGET_BOX = (int(TRACK_W/2) - 340, int(TRACK_H/2) - 220,
        int(TRACK_W/2) + 340, int(TRACK_H/2) + 220)

FACE_COUNT = 0

CORRECTION_THRESHOLD = 0.50

class DaisyEye:
    cam = None
    known_faces = {}
    data_queue = None
    flipped = False

    def __init__(self, faces, data_queue = None, cam_num = 1,
            res = (FACE_W, FACE_H), flipped = False):
        self.cam = cv2.VideoCapture(cam_num);

        if not self.cam.isOpened():
            print("Could not open camera...")
            sys.exit()

        for person in faces:
            image = face_recognition.load_image_file(faces[person])
            print(person)
            face_encoding_list = face_recognition.face_encodings(image)
            if len(face_encoding_list) > 0:
                self.known_faces[person] = face_encoding_list[0]
            else:
                print("\tCould not find face for person...")

        self.cam.set(3, res[0])
        self.cam.set(4, res[1])
        self.cam.set(14, EXPOSURE_2)

        self.data_queue = data_queue
        self.flipped = flipped

    def __draw_bbox(self, valid, frame, bbox, color, text):
        if not valid:
            return
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2, 1)
        cv2.putText(frame, text, (bbox[0], bbox[1] - 4), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def __scale_frame(self, frame, scale_factor = 1):
        if scale_factor == 1:
            return frame
        return cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)

    def __crop_frame(self, frame, crop_box):
        return frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2],:].copy()

    def __init_tracker(self, frame, bbox, tracker_type = "BOOSTING"):
        tracker = None;

        print("Init Tracker with:", bbox, tracker_type)

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

    def view(self, face_bbox = DEFAULT_FACE_TARGET_BOX, bbox_list = []):
        print("Press q when image is ready")
        while True:
            _, frame = self.cam.read()
            if self.flipped:
                frame = cv2.flip(frame, 0)
            output_frame = frame.copy()

            self.__draw_bbox(ret, output_frame, face_bbox, (255,0,0), "Target")
            count = 0
            for bbox in bbox_list:
                self.__draw_bbox(ret, output_frame, bbox, (255, 0, 0), str(count))
                print(count, self.__bbox_overlap(face_bbox, bbox), self.__bbox_overlap(bbox, face_bbox))
                count += 1
            cv2.imshow("Eye View", output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        return ret, frame

    def __select_ROI(self, frame):
        bbox = cv2.selectROI(frame, False)
        cv2.destroyAllWindows()
        return bbox;

    def release_cam(self):
        cv2.destroyAllWindows()
        self.cam.release()

    """
    Scale from res1 to res2
    """
    def __scale_bbox(self, bbox, res1, res2):
        scaled = (int(bbox[0] * res2[0] / res1[0]), \
                int(bbox[1] * res2[1] / res1[1]), \
                int(bbox[2] * res2[0] / res1[0]), \
                int(bbox[3] * res2[1] / res1[1]))
        return scaled

    """
    Standard bbox layout (left, top, right, bottom)
    bbox1 overlaps with bbox2?
    """
    def __bbox_overlap(self, bbox1, bbox2):
        if not bbox1 or not bbox2:
            return 0

        left = max(bbox1[0], bbox2[0])
        top = max(bbox1[1], bbox2[1])
        right = min(bbox1[2], bbox2[2])
        bottom = min(bbox1[3], bbox2[3])

        if left < right and top < bottom:
            return self.__bbox_area((left, top, right, bottom))
        return 0

    def __bbox_area(self, bbox):
        if not bbox:
            return 0
        (left, top, right, bottom) = bbox
        return (right - left) * (bottom - top)

    def find_and_track_correcting(self, name, tracker = "CSRT",
            track_target_box = DEFAULT_TRACK_TARGET_BOX,
            face_target_box = DEFAULT_FACE_TARGET_BOX,
            res = (FACE_W, FACE_H),
            video_out = True, debug = True):
        print("Finding and Tracking with Correction")

        trackerObj = None

        self.cam.set(3, res[0])
        self.cam.set(4, res[1])
        self.cam.set(14, EXPOSURE_1)

        face_count = 5
        face_process_frame = True

        bbox = None
        face_bbox = None
        track_bbox = None

        while True:
            _, frame = self.cam.read()
            if self.flipped:
                frame = cv2.flip(frame, 0)

            timer = cv2.getTickCount()

            person_found = False

            small_frame = self.__crop_frame(frame, face_target_box)

            if face_process_frame:
                face_locations = face_recognition.face_locations(
                        small_frame, model="cnn")
                face_encodings = face_recognition.face_encodings(
                        small_frame, face_locations)


                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                            [self.known_faces[name]], face_encoding, 0.6)

                    if len(matches) > 0 and matches[0]:
                        person_found = True

                        face_count += 1

                        (top, right, bottom, left) = face_locations[0]

                        left += face_target_box[0]
                        top += face_target_box[1]
                        right += face_target_box[0]
                        bottom += face_target_box[1]

                        face_bbox = (left, top, right, bottom)

            face_process_frame = not face_process_frame
            status = False

            overlap_pct = 0
            track_area = self.__bbox_area(track_bbox)
            if track_area > 0 and face_bbox:
                overlap_area = self.__bbox_overlap(face_bbox, track_bbox)
                overlap_pct = min(overlap_area / self.__bbox_area(face_bbox), \
                        overlap_area / self.__bbox_area(track_bbox))

            small_frame = self.__crop_frame(frame, track_target_box)

            if person_found and face_count >= FACE_COUNT and overlap_pct < CORRECTION_THRESHOLD:
                # Re-init tracker
                bbox = (face_bbox[0] - track_target_box[0], \
                        face_bbox[1] - track_target_box[1], \
                        face_bbox[2] - face_bbox[0], \
                        face_bbox[3] - face_bbox[1])
                trackerObj = self.__init_tracker(small_frame, bbox, tracker)
                face_count = 0

            if trackerObj is not None:
                status = False

                if tracker == "DLIB":
                    status = trackerObj.update(small_frame)
                    rect = trackerObj.get_position()
                    bbox = (int(rect.left()), int(rect.top()), \
                            int(rect.right()), int(rect.bottom()))
                else:
                    status, trackerBBox = trackerObj.update(small_frame)
                    bbox = (int(trackerBBox[0]), \
                            int(trackerBBox[1]), \
                            int(trackerBBox[0] + trackerBBox[2]), \
                            int(trackerBBox[1] + trackerBBox[3]))
            if bbox is not None:
                track_bbox = (bbox[0] + track_target_box[0], \
                        bbox[1] + track_target_box[1], \
                        bbox[2] + track_target_box[0], \
                        bbox[3] + track_target_box[1])

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

            if status:
                self.__update_individual_position("NONE", track_bbox, -1, res)

            if video_out:
                cv2.line(frame, (0, int(res[1]/2)), \
                        (int(res[0]), int(res[1]/2)), (255,0,0), 1)
                cv2.line(frame, (int(res[0]/2), 0),
                        (int(res[0]/2), int(res[1])), (255,0,0), 1)

                self.__draw_bbox(True, frame, track_target_box, (255, 0, 0), "TRACK_TARGET")

                self.__draw_bbox(True, frame, face_target_box, (255, 0, 0), "FACE_TARGET")

                self.__draw_bbox(status, frame, track_bbox, (0, 255, 0), tracker)

                self.__draw_bbox(person_found, frame, face_bbox, (0, 0, 255), name)

                cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

                failedTrackers = "FAILED: "
                if not status:
                    failedTrackers += tracker + " "
                cv2.putText(frame, failedTrackers, (100, 80), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,142), 1)

                frame = self.__scale_frame(frame, scale_factor=0.50)

                cv2.imshow("Daisy's Vision", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Interrupted")
                    cv2.destroyAllWindows()
                    return 0

            if debug:
                print(fps, track_bbox, face_bbox)

        cv2.destroyAllWindows()

    def __update_individual_position(self, str_pos, track_bbox, distance, res):
        if self.data_queue is not None and self.data_queue.empty():
            self.data_queue.put((str_pos, track_bbox, distance, res))
