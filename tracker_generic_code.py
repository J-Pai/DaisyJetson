import cv2

CAM_NUM = 1

def track_object():
    video = cv2.VideoCapture(CAM_NUM)           # Setup the input video

    video.set(3, 640)
    video.set(4, 480)

    ok, frame = video.read()
    tracker = cv2.TrackerKCF_create()           # Create the tracker object
    bbox = cv2.selectROI(frame, False)          # Select the desired object to track
    ok = tracker.init(frame, bbox)              # Initialize tracker with bbox and starting frame
    while True:

        timer = cv2.getTickCount()

        _, frame = video.read()
        ok, bbox = tracker.update(frame)        # Update tracker with new frame to obtain new bbox
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

        cv2.imshow("Tracking", frame)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break

track_object()

