import cv2
import face_recognition
import sys

class DaisyEye:
    cam = None
    scale_factor = 0
    known_faces = {}
    frame = None
    output_frame = None

    def __init__(self, faces, cam_num = 1, scale_factor = 1, res_width = 1024, res_height = 576):
        cv2.setNumThreads(100)

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

        self.scale_factor = scale_factor

        self.cam.set(3, res_width)
        self.cam.set(4, res_height)

    def draw_bbox(self, valid, frame, bbox, color, text):
        if not valid:
            return
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1, 1)
        cv2.putText(frame, text, (int(bbox[0]), int(bbox[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def scale_frame(self, frame):
        if self.scale_factor == 1:
            return frame
        return cv2.resize(frame, (0,0), fx=self.scale_factor, fy=self.scale_factor)

    def locate_target(self, name, ret = False, video_out = True, debug = True):
        print("Start Locating Target: " + name, ret, video_out, debug)
        face_locations = []
        face_encodings = []

        bbox = None

        while True:
            valid, self.frame = self.cam.read()
            if not valid:
                print("Failure to read camera")
                return -1
            if video_out:
                self.output_frame = self.frame[:,:,:].copy()
            rgb_small_frame = self.scale_frame(self.frame)
            timer = cv2.getTickCount()

            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            person_found = False

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces([self.known_faces[name]], face_encoding, 0.6)

                if len(matches) > 0 and matches[0]:
                    person_found = True

                    (top, right, bottom, left) = face_locations[0]

                    left *= int(1/self.scale_factor)
                    top *= int(1/self.scale_factor)
                    right *= int(1/self.scale_factor)
                    bottom *= int(1/self.scale_factor)

                    bbox = (left, top, right, bottom)

                    if video_out:
                        self.draw_bbox(valid, self.output_frame, bbox, (0, 0, 255), name)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            if video_out:
                cv2.putText(self.output_frame, "FPS: " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Locate_Target", self.output_frame)

            if debug:
                print(fps, face_locations)

            if ret and person_found:
                cv2.destroyAllWindows()
                print("Done Locating Target")
                return bbox

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                print("Interrupted")
                return 0

        cv2.destroyAllWindows()

    def init_tracker(self, frame, bbox, tracker_type = "BOOSTING"):
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

    def select_ROI(self, frame):
        bbox = cv2.selectROI(frame, False)
        cv2.destroyAllWindows()
        return bbox;


    def track_object(self, obj = None, tracker = "CSRT", color = (0,255,0), video_out = True, debug = True):
        print("Start Tracking Obj", video_out, debug)

        bbox = obj
        if bbox is None:
            valid, self.frame = self.cam.read()
            if video_out:
                self.output_frame = self.frame[:,:,:].copy()
            if not valid:
                print("Failure to read camera")
                return -1
            bbox = self.select_ROI(self.frame)
        else:
            bbox = (obj[0], obj[1], obj[2] - obj[0], obj[3] - obj[1])



        trackerObj = self.init_tracker(self.frame, bbox, tracker)

        bbox = None

        while True:
            valid, self.frame = self.cam.read()
            if video_out:
                self.output_frame = self.frame[:,:,:].copy()
            if not valid:
                print("Failure to read camera")
                return -1

            timer = cv2.getTickCount()

            tracker_ret_and_bbox = trackerObj.update(self.frame)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

            trackerBBox = tracker_ret_and_bbox[1]

            left = trackerBBox[0]
            top = trackerBBox[1]
            right = trackerBBox[0] + trackerBBox[2]
            bot = trackerBBox[1] + trackerBBox[3]

            bbox = (left, top, right, bot)
            if video_out:
                self.draw_bbox(tracker_ret_and_bbox[0], \
                    self.output_frame, \
                    bbox, \
                    color,
                    tracker)

            if video_out:
                cv2.putText(self.output_frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                failedTrackers = "FAILED: "
                if not tracker_ret_and_bbox[0]:
                    failedTrackers += tracker + " "
                cv2.putText(self.output_frame, failedTrackers, (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,142), 1)

                cv2.imshow("Track_Object", self.output_frame)

            if debug:
                print(fps, bbox)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Interrupted")
                cv2.destroyAllWindows()
                return 0

        cv2.destroyAllWindows()

    def release_cam(self):
        cv2.destroyAllWindows()
        self.cam.release()

    def find_and_track(self, name, video = True, dbg = True):
        print("Finding and Tracking...")
        init_bbox = self.locate_target("JessePai", ret = True, video_out = video, debug = dbg)
        if type(init_bbox) is not tuple:
            sys.exit()
        print(init_bbox)
        self.track_object(init_bbox, video_out = video, debug = dbg)
        cv2.destroyAllWindows()
        self.cam.release()
        print("Done!")

faces = {
    "JessePai": "./faces/JPai-2.png",
}

if __name__ == "__main__":
    eye = DaisyEye(faces)
    init_bbox = eye.locate_target("JessePai", ret = True, debug = False)
    if type(init_bbox) is not tuple:
        sys.exit()
    print(init_bbox)
    eye.track_object(init_bbox, debug = False)
