import cv2
import time
from vortex.detector.vehicle_detector import VehicleDetector
from vortex.tracker.sort import Sort
from vortex.traffic.utils import parse_cfg


class VideoAnalyzer(object):
    def __init__(self, cfg_file):
        self.cfg = parse_cfg(cfg_file)
        self.video_cap = None

        # initialize objects
        self.vehicle_detector = VehicleDetector(self.cfg['vehicle']['model_path'], 
                                                self.cfg['vehicle']['label_file'])
        self.mot_tracker = Sort()
        
    def open(self, video_url):
        if self.video_cap is not None:
            self.video_cap.release()
        self.video_cap = cv2.VideoCapture(video_url)
        if not self.video_cap.isOpened():
            return False
        self.video_fps = float(self.video_cap.get(cv2.VIDEO_CAP_PROP_FPS))
        self.video_width = int(self.video_cap.get(cv2.VIDEO_CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video_cap.get(cv2.VIDEO_CAP_PROP_FRAME_HEIGHT))
        return True

    def run(self, display=False):
        while True:
            ret, image = self.video_cap.read()
            if not ret:
                break
            boxes = self.vehicle_detector.detect(image)
            print(boxes)
