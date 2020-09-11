"""
test kcf single object tracker
"""
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from vortex.tracker.kcftracker import KCFTracker
import time


class InitBoxDrawer(object):
    def __init__(self):
        self.ix = -1
        self.iy = -1
        self.cx = -1
        self.cy = -1
        self.min_gap = 10
        self.selecting_obj = False
        self.selected = False
    
    def select(self, image):
        cv2.namedWindow('box selection')
        cv2.setMouseCallback('box selection', self.mouse_callback)
        while not self.selected:
            # draw box and show
            image2 = image.copy()
            if self.selecting_obj:
                cv2.rectangle(image2, (self.ix, self.iy), (self.cx, self.cy), (0, 255, 0), 1)
            cv2.imshow('box selection', image2)
            cv2.waitKey(27)
        cv2.destroyAllWindows()
        return [self.ix, self.iy, self.cx, self.cy]

    def mouse_callback(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting_obj = True
            self.ix = x
            self.iy = y
        elif event == cv2.EVENT_MOUSEMOVE:
            self.cx = x
            self.cy = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting_obj = False
            w, h = abs(x - self.ix), abs(y - self.iy)
            if w > self.min_gap and h > self.min_gap:
                if self.ix > self.cx:
                    self.ix, self.cx = self.cx, self.ix
                if self.iy > self.cy:
                    self.iy, self.cy = self.cy, self.iy
                self.selected = True


class KCFTrackingTester(object):
    def __init__(self):
        self.tracker = KCFTracker(True, True, True)

    def minmax2xywh(self, box):
        return [box[0], box[1], box[2] - box[0], box[3] - box[1]]

    def xywhwminmax(self, box):
        return [box[0], box[1], box[0] + box[2], box[1] + box[3]]

    def test_video(self, video_path, start_frame=0):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print('cannot open video {}'.format(video_path))
            return
        # skip to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # read initial frame
        ret, init_frame = cap.read()
        if not ret:
            print('cannot read frame')
            return
        
        # draw init box
        box_drawer = InitBoxDrawer()
        init_box = box_drawer.select(init_frame)
        print(init_box)  

        # start tracking
        start_time = time.time()
        # this version of kcf task long time when initializing
        self.tracker.init(self.minmax2xywh(init_box), init_frame)
        end_time = time.time()
        print('tracker initialization time: ', end_time - start_time)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            box = self.tracker.update(frame)
            box = self.xywhwminmax(box)
            box = [int(v) for v in box]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imshow('tracking', frame)
            cv2.waitKey(25)

        
if __name__ == '__main__':
    video_path = '../../samples/zhroad_surveillance.avi'
    tester = KCFTrackingTester()
    tester.test_video(video_path, start_frame=0)
