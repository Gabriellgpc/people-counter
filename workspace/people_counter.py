import numpy as np
import click
import time
import dlib
import cv2

from src.centroid_tracker import CentroidTracker
from src.trackable_object import TrackableObject
from src.helper import imresize

HEIGHT = 500

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def run_detector(image, network, min_confidence):
    h, w = image.shape[:2]
    # preprocess the image
    preproc_image = cv2.resize(image,dsize=[300,300])
    preproc_image = cv2.cvtColor(preproc_image, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(preproc_image, 0.007843, (300,300), 127.5)
    network.setInput(blob)
    detections = network.forward()

    valid_detections = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > min_confidence:
            class_idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            valid_detections.append( (startX, startY, endX, endY, class_idx) )
    return valid_detections

@click.command()
@click.option('-p', '--prototxt',
    default='/home/condados/workarea/people-couting/assets/public/mobilenet-ssd/caffe/mobilenet-ssd.prototxt',
    help='Path to Caffe prototxt file of the trained model')
@click.option('-m','--model',
    default='/home/condados/workarea/people-couting/assets/public/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel',
    help='Path to Caffe model')
@click.option('-i','--input-video',
    default='/home/condados/workarea/people-couting/data/input_sample/video-stadium.mp4',
    help='Path to input video file')
@click.option('-o','--output-video',
    default='/home/condados/workarea/people-couting/data/output_sample/demo.mp4',
    help='Path to output video file')
@click.option('-c','--confidence', type=float, default=0.4,
    help='To filter weak detection')
@click.option('-s','--skip-frames', type=int, default=30,
    help='# of frames to skip between detections')
def main(prototxt, model, input_video, output_video, confidence, skip_frames):
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    ct = CentroidTracker()
    trackers = []
    trackableObjects = {}
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    roi = None
    video = cv2.VideoCapture(input_video)
    # video = cv2.VideoCapture('/home/condados/workarea/people-couting/data/input_sample/05.mp4')
    # video = cv2.VideoCapture('/dev/video0')
    # time.sleep(2.0)
    while True:
        res, frame = video.read()

        assert res == True, 'Error while reading video from or the file has ended'
        if roi is None:
            roi = cv2.selectROI('select ROI', frame)
        # roi => startX, startY, endX, endY
        frame = frame[roi[1]:roi[-1],roi[0]:roi[-2],:].copy()
        frame = imresize(frame, desired_height=HEIGHT, keep_aspect=True)
        h, w = frame.shape[:2]
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        status = 'Waiting'
        rects = []
        if totalFrames % skip_frames == 0:
            status = 'Detecting'
            trackers = []
            # model inference
            detections = run_detector(frame, net, min_confidence=confidence)

            # draw detections
            image = frame.copy()
            for i in range(len(detections)):
                startX, startY, endX, endY, class_idx = detections[i]
                label = CLASSES[class_idx]
                if label != 'person':
                    continue
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track( frame_RGB, rect )

                trackers.append(tracker)
                rects.append( [startX, startY, endX, endY] )

                # Visualization
                # cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[class_idx], 2)
                # y = startY - 15 if startY - 15 > 15 else startY + 15
                # cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_idx], 2)
        # otherwise, we should utilize our object *trackers* rather than
	    # object *detectors* to obtain a higher frame processing throughput
        else:
            for tracker in trackers:
                status = 'Tracking'
                tracker.update(frame_RGB)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endX = int(pos.bottom())
                rects.append( [startX, startY, endX, endY] )

        # a horizontal visualization line
        # (that people must cross in order to be tracked)
        # and use the centroid tracker to update our object centroids
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(image, (0, h // 2), (w, h // 2), (0, 255, 255), 2)

        # use the centroid tracker to associate the (1) old object
	    # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        for objectID, centroid in objects.items():
            # check to see if a trackable object exists for the current
		    # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
		    # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

			    # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < h//2:
                        totalUp += 1
                        to.counted = True
                    elif direction > 0 and centroid[1] > h//2:
                        totalDown += 1
                        to.counted = True
            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            text = 'ID {}'.format(objectID)
            cv2.putText(image,
                        text,
                        (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0,), -1)

            # construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Up", totalUp),
            ("Down", totalDown),
            ("Status", status),
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(image, text, (10, h - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)



        cv2.imshow('video', image)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break
    # close any open windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()