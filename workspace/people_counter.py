import numpy as np
import click
import time
import dlib
import cv2

from src.centroid_tracker import CentroidTracker
from src.trackable_object import TrackableObject
from src.helper import imresize

HEIGHT = 480

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

    video = cv2.VideoCapture(input_video)
    # video = cv2.VideoCapture('/home/condados/workarea/people-couting/data/input_sample/05.mp4')
    # video = cv2.VideoCapture('/dev/video0')
    # time.sleep(2.0)
    while True:
        res, frame = video.read()

        assert res == True, 'Error while reading video from or the file has ended'
        frame = imresize(frame, desired_height=HEIGHT, keep_aspect=True)

        # model inference
        detections = run_detector(frame, net, min_confidence=confidence)

        # draw detections
        image = frame
        bboxes = []
        for i in range(len(detections)):
            startX, startY, endX, endY, class_idx = detections[i]
            label = CLASSES[class_idx]
            if label != 'person':
                continue
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[class_idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_idx], 2)
            bboxes.append( [startX, startY, endX, endY] )

        objects = ct.update(bboxes)
        for objectID, centroid in objects.items():
            text = 'ID {}'.format(objectID)
            print('Centroid {}'.format(centroid))
            cv2.putText(image,
                        text,
                        (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0,), -1)

        cv2.imshow('video', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break


if __name__ == '__main__':
    main()