"""
Test sort and sort_with_labels
"""
import cv2
import numpy as np
from collections import defaultdict
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from vortex.tracker.sort import Sort
from vortex.tracker.sort_with_labels import Sort as Sort2

class SortTester(object):
    def __init__(self):
        pass

    def test_sort(self, dets_file):
        frames_dets = self.read_dets(dets_file)
        sort = Sort(max_age=2, min_hits=2)
        num_frames = max(frames_dets.keys())
        for i in range(1, num_frames+1):
            dets = np.array(frames_dets[i])
            tracks = sort.update(dets)
            # print(tracks)
            img = self.get_image(tracks)
            cv2.imshow('image', img)
            cv2.waitKey(25)

    def test_sort_with_labels(self, dets_file):
        frames_dets = self.read_dets(dets_file)
        sort = Sort2(max_age=5, min_hits=1)
        num_frames = max(frames_dets.keys())
        for i in range(1, num_frames+1):
            dets = np.array(frames_dets[i])
            labels = [0 for _ in range(dets.shape[0])]
            tracks = sort.update(dets, labels)
            # print(tracks)
            img = self.get_image(tracks)
            cv2.imshow('image', img)
            cv2.waitKey(25)

    def read_dets(self, dets_file):
        frames = defaultdict(list)
        prev_index = 0
        with open(dets_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split(',')
                frame_idx = int(parts[0])
                box = [float(v) for v in parts[2:7]]
                box[2] += box[0]
                box[3] += box[1]
                frames[frame_idx].append(box)
        return frames
    
    def get_image(self, tracks, image=None, image_width=1920, image_height=1080):
        if image is None:
            image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        ntracks = tracks.shape[0]
        for t in range(ntracks):
            xmin = int(tracks[t, 0])
            ymin = int(tracks[t, 1])
            xmax = int(tracks[t, 2])
            ymax = int(tracks[t, 3])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            idx = int(tracks[t, 4])
            if tracks.shape[1] == 6:
                label = int(tracks[t, 5])
                text = '{}-{}'.format(idx, label)
            else:
                text = '{}'.format(idx)
            
            cv2.putText(image, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return image


if __name__ == '__main__':
    tester = SortTester()
    filepath = '../samples/det.txt'
    # tester.test_sort(filepath)
    tester.test_sort_with_labels(filepath)
