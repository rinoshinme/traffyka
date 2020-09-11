"""
test simple mot with opencv tracker
"""
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from vortex.detector.vehicle_detector import VehicleDetector
# from vortex.detector.yolov3_detector import Yolov3Detector
from vortex.tracker.simple_mot_tracker import SimpleMotTracker


class SimpleMotTrackerTester(object):
    def __init__(self, *detector_params):
        # self.detector = Yolov3Detector(*detector_params)
        self.detector = VehicleDetector(*detector_params)
        self.tracker = SimpleMotTracker()

    def visualize(self, image, detections=None, trackings=None):
        color_detection = (255, 255, 255)
        color_tracking = (0, 255, 0)
        image = image.copy()
        if detections is not None:
            ndetections = detections.shape[0]
            for i in range(ndetections):
                xmin = int(detections[i, 0])
                ymin = int(detections[i, 1])
                xmax = int(detections[i, 2])
                ymax = int(detections[i, 3])
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_detection, 1)

        if trackings is not None:
            ntrackings = trackings.shape[0]
            for i in range(ntrackings):
                xmin = int(trackings[i, 0])
                ymin = int(trackings[i, 1])
                xmax = int(trackings[i, 2])
                ymax = int(trackings[i, 3])
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_tracking, 1)
        return image

    def test_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print('cannot open video: {}'.format(video_path))
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # do detection
            boxes, scores, labels = self.detector.detect(frame)
            print('detections:')
            print(boxes)
            tracks = self.tracker.update(frame, boxes)
            print('tracks')
            print(tracks)
            image = self.visualize(frame, boxes, tracks)
            cv2.imshow('tracking', image)
            cv2.waitKey(1)


if __name__ == '__main__':
    # yolo_cfg = '../../model_data/yolov3.cfg'
    # yolo_weights = '../../model_data/yolov3.weights'
    onnx_path = '../../model_data/yolo_vehicle.onnx'
    label_file = '../../model_data/vehicle.names'
    tester = SimpleMotTrackerTester(onnx_path, label_file)
    video_path = '../../samples/highway_traffic.mp4'
    tester.test_video(video_path)
