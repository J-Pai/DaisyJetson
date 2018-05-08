import numpy as np
import cv2
import face_recognition
import sys

from multiprocessing import Queue
from multiprocessing.managers import SyncManager
from queue import Queue as ImageQueue
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import setGlobalLogger

setGlobalLogger(None)
print("OpenGL Pipeline")
from pylibfreenect2 import OpenGLPacketPipeline

RGB_W = 1920
RGB_H = 1080

FACE_W = RGB_W
FACE_H = RGB_H
DEFAULT_FACE_TARGET_BOX = (int(RGB_W/2) - 125, int(RGB_H/2) - 125,
        int(RGB_W/2) + 125, int(RGB_H/2) + 125)

DEFAULT_SCALING = 0.25

FACE_COUNT = 0

CORRECTION_THRESHOLD = 0.50

class NeuronManager(SyncManager):
    pass

NeuronManager.register('get_web_neuron')
NeuronManager.register('get_alexa_neuron')

class DaisyEye:
    cam = None
    known_faces = {}
    data_queue = None
    pipeline = None
    connected = True
    manager = None
    web_neuron = None
    alexa_neuron = None

    def __init__(self, faces, data_queue = None):
        for person in faces:
            image = face_recognition.load_image_file(faces[person])
            face_encoding_list = face_recognition.face_encodings(image)
            if len(face_encoding_list) > 0:
                self.known_faces[person] = face_encoding_list[0]
            else:
                print("\tCould not find face for person...")

        self.data_queue = data_queue
        self.pipeline = OpenGLPacketPipeline()

        self.manager = NeuronManager(address=('', 4081), authkey=b'daisy')
        try:
            self.manager.connect()
            self.web_neuron = self.manager.get_web_neuron()
            self.web_neuron.clear()
            self.alexa_neuron = self.manager.get_alexa_neuron()
            print("Eye connected to neuron manager.")
        except ConnectionRefusedError:
            print("Eye not connected to neuron manager.")
            self.connected = False

    def __draw_bbox(self, valid, frame, bbox, color, text):
        if not valid:
            return
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2, 1)
        cv2.putText(frame, text, (bbox[0], bbox[1] - 4), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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

    """
    Scale from res1 to res2
    """
    def __scale_bbox(self, bbox, scale_factor = 1):
        scaled = (int(bbox[0] * scale_factor), \
                int(bbox[1] * scale_factor), \
                int(bbox[2] * scale_factor), \
                int(bbox[3] * scale_factor))
        return scaled

    def __scale_frame(self, frame, scale_factor = 1):
        if scale_factor == 1:
            return frame
        return cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)

    def __crop_frame(self, frame, crop_box):
        return frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2],:].copy()

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

    def __body_bbox(self, bigdepth, mid_width, mid_height, res):
        mid_dist = bigdepth[mid_height][mid_width]

        mid_col = bigdepth[:, mid_width]

        farther_pixels = np.argwhere(mid_col > mid_dist)

        lower_bound = (farther_pixels + 1)[:-1]
        upper_bound = (farther_pixels - 1)[1:]
        mask = lower_bound <= upper_bound

        upper_bound, lower_bound = upper_bound[mask], lower_bound[mask]

        adjust = 100
        top_of_head = mid_height
        if len(lower_bound) > 0:
            top_of_head = lower_bound[0]
        top_of_head = 1 if top_of_head - adjust < 1 else top_of_head - adjust

        body_mid_height = int((top_of_head + res[1])/2)
        mid_row = bigdepth[body_mid_height, :]
        body_mid_dist = bigdepth[body_mid_height][mid_width]

        farther_pixels = np.argwhere(mid_row > body_mid_dist + 150)

        lower_bound = (farther_pixels + 1)[:-1]
        upper_bound = (farther_pixels - 1)[1:]
        mask = lower_bound <= upper_bound

        upper_bound, lower_bound = upper_bound[mask], lower_bound[mask]

        target = None
        for i in range(0, len(upper_bound)):
            if upper_bound[i] >= mid_width and lower_bound[i] <= mid_width:
                target = i
                break

        left = lower_bound[target]
        top = top_of_head
        right = upper_bound[target]
        bottom = res[1]

        return (left, top, right, bottom)

    def find_and_track_kinect(self, name, tracker = "CSRT",
            face_target_box = DEFAULT_FACE_TARGET_BOX,
            track_scaling = DEFAULT_SCALING,
            res = (RGB_W, RGB_H), video_out = True):
        print("Starting Tracking")

        target = name

        fn = Freenect2()
        num_devices = fn.enumerateDevices()

        if num_devices == 0:
            print("No device connected!")

        serial = fn.getDeviceSerialNumber(0)
        device = fn.openDevice(serial, pipeline = self.pipeline)

        listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth)

        device.setColorFrameListener(listener)
        device.setIrAndDepthFrameListener(listener)

        device.start()

        registration = Registration(device.getIrCameraParams(),
                device.getColorCameraParams())

        undistorted = Frame(512, 424, 4)
        registered = Frame(512, 424, 4)
        bigdepth = Frame(1920, 1082, 4)

        trackerObj = None
        face_process_frame = True

        bbox = None
        track_bbox = None

        head_h = 0
        body_left_w = 0
        body_right_w = 0
        center_w = 0

        globalState = ""

        # Following line creates an avi video stream of daisy's tracking
        # out = cv2.VideoWriter('daisy_eye.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (960, 540))
        # out.write(c)
        while True:
            timer = cv2.getTickCount()

            frames = listener.waitForNewFrame()

            color = frames["color"]
            depth = frames["depth"]

            registration.apply(color, depth, undistorted, registered, bigdepth=bigdepth)

            bd = np.resize(bigdepth.asarray(np.float32), (1080, 1920))
            c = cv2.cvtColor(color.asarray(), cv2.COLOR_RGB2BGR)

            if self.connected:
                newTarget = self.alexa_neuron.get('name');
                if newTarget != target:
                    target = newTarget
                    listener.release(frames)
                    trackerObj = None
                    face_process_frame = True

                    bbox = None
                    track_bbox = None
                    continue
                if target is not None and target not in self.known_faces:
                    target = None

            if target is None:
                if self.connected:
                    c = self.__scale_frame(c, scale_factor = 0.5)
                    image = cv2.imencode('.jpg', c)[1].tostring()
                    self.web_neuron.update([('image', image)])
                listener.release(frames)
                trackerObj = None
                face_process_frame = True

                bbox = None
                track_bbox = None
                self.__update_individual_position("WAITING", None, None, None, res)
                continue

            face_bbox = None
            new_track_bbox = None

            if face_process_frame:
                small_c = self.__crop_frame(c, face_target_box)
                face_locations = face_recognition.face_locations(small_c, number_of_times_to_upsample=0, model="cnn")
                face_encodings = face_recognition.face_encodings(small_c, face_locations)
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                            [self.known_faces[target]], face_encoding, 0.6)
                    if len(matches) > 0 and matches[0]:
                        (top, right, bottom, left) = face_locations[0]

                        left += face_target_box[0]
                        top += face_target_box[1]
                        right += face_target_box[0]
                        bottom += face_target_box[1]

                        face_bbox = (left, top, right, bottom)
                        mid_w = int((left + right) / 2)
                        mid_h = int((top + bottom) / 2)
                        new_track_bbox = self.__body_bbox(bd, mid_w, mid_h, res)

                        break
            face_process_frame = not face_process_frame

            overlap_pct = 0
            track_area = self.__bbox_area(track_bbox)
            if track_area > 0 and new_track_bbox:
                overlap_area = self.__bbox_overlap(new_track_bbox, track_bbox)
                overlap_pct = min(overlap_area / self.__bbox_area(new_track_bbox),
                        overlap_area / self.__bbox_area(track_bbox))
            small_c = self.__scale_frame(c, track_scaling)
            if new_track_bbox is not None and overlap_pct < CORRECTION_THRESHOLD:
                bbox = (new_track_bbox[0],
                        new_track_bbox[1],
                        new_track_bbox[2] - new_track_bbox[0],
                        new_track_bbox[3] - new_track_bbox[1])
                bbox = self.__scale_bbox(bbox, track_scaling)
                trackerObj = self.__init_tracker(small_c, bbox, tracker)
                self.alexa_neuron.update([('tracking', True)])

            if trackerObj is None:
                self.__update_individual_position("WAITING", None, None, None, res)

            status = False

            if trackerObj is not None:
                status, trackerBBox = trackerObj.update(small_c)
                bbox = (int(trackerBBox[0]),
                        int(trackerBBox[1]),
                        int(trackerBBox[0] + trackerBBox[2]),
                        int(trackerBBox[1] + trackerBBox[3]))

            if bbox is not None:
                track_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
                track_bbox = self.__scale_bbox(bbox, 1/track_scaling)

            w = 0
            h = 0

            if status:
                w = track_bbox[0] + int((track_bbox[2] - track_bbox[0])/2)
                h = track_bbox[1] + int((track_bbox[3] - track_bbox[1])/2)

                if (w < res[0] and w >= 0 and h < res[1] and h >= 0):
                    distanceAtCenter =  bd[h][w]
                    center = (w, h)
                    globalState = self.__update_individual_position(status, track_bbox, center, distanceAtCenter, res)



            if globalState == "Fail":
                break

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            if video_out or self.connected:
                cv2.line(c, (w, 0), (w, res[1]), (0,255,0), 1)
                cv2.line(c, (0, h), (res[0], h), (0,255,0), 1)
                cv2.line(c, (0, head_h), (res[0], head_h), (0,0,0), 1)
                cv2.line(c, (body_left_w, 0), (body_left_w, res[1]), (0,0,255), 1)
                cv2.line(c, (body_right_w, 0), (body_right_w, res[1]), (0,0,255), 1)
                cv2.line(c, (center_w, 0), (center_w, res[1]), (0,0,255), 1)

                self.__draw_bbox(True, c, face_target_box, (255, 0, 0), "FACE_TARGET")
                self.__draw_bbox(status, c, track_bbox, (0, 255, 0), tracker)
                self.__draw_bbox(face_bbox is not None, c, face_bbox, (0, 0, 255), target)
                self.__draw_bbox(face_bbox is not None, c, new_track_bbox, (0, 255, 255), "BODY")

                c = self.__scale_frame(c, scale_factor = 0.5)

                cv2.putText(c, "FPS : " + str(int(fps)), (100,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                if not status:
                    failedTrackers = "FAILED: "
                    failedTrackers += tracker + " "
                    cv2.putText(c, failedTrackers, (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,142), 1)
                if self.connected:
                    image = cv2.imencode('.jpg', c)[1].tostring()
                    self.web_neuron.update([('image', image)])
                if video_out:
                    cv2.imshow("color", c)

            listener.release(frames)

            key = cv2.waitKey(1) & 0xff
            if key == ord('q'):
                self.__update_individual_position("STOP", None, None, None, res)
                break

        self.so.close()
        cv2.destroyAllWindows()
        device.stop()
        device.close()

    def __update_individual_position(self, status, track_bbox, center, distance, res):
        if self.data_queue is None:
            return "Fail"
        if self.data_queue.empty():
            self.data_queue.put((status, track_bbox, center, distance, res))
            return "Success"
