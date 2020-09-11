"""
Multiple object tracking with 
    object detection and 
    opencv-kcf [or other opencv trackers]
"""

import cv2
import numpy as np


class TrackerObject(object):
    OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}
    count = 0
    def __init__(self, init_box, init_frame, tracker_type='kcf'):
        self.tracker = TrackerObject.OPENCV_OBJECT_TRACKERS[tracker_type]()
        self.init_tracker(init_box, init_frame)
        self.id = TrackerObject.count
        TrackerObject.count += 1
    
    def init_tracker(self, init_box, init_frame):
        self.box = init_box
        self.tracker.init(init_frame, self.minmax2xywh(init_box))
        self.time_since_update = 0

    def update(self, frame):
        _, box = self.tracker.update(frame)
        self.box = self.xywh2minmax(box)
        self.time_since_update += 1
    
    def update_with_box(self, box, frame):
        """
        if a tracker is matched with a detection box, 
        the box in the tracker is directly replaced by the detection box.
        TODO: interpolate between tracker box and detection box.
        """
        # reinitialize the tracker.
        self.init_tracker(box, frame)

    def minmax2xywh(self, box):
        return (box[0], box[1], box[2] - box[0], box[3] - box[1])

    def xywh2minmax(self, box):
        return (box[0], box[1], box[0] + box[2], box[1] + box[3])
    
    def get_bbox(self):
        return self.box


class SimpleMotTracker(object):
    """
    or SMT for short.
    """
    def __init__(self, max_age=5, iou_threshold=0.6):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.trackers = []

    def update(self, frame, dets=None):
        # update trackers
        for trk in self.trackers:
            trk.update(frame)
        # calculate iou for trackers and detections
        tracks = np.array([trk.get_bbox() for trk in self.trackers])
        matches, unmatched_detections, unmatched_tracks = self.associate_detections_to_trackers(dets, tracks)
        # update matches
        for m in matches:
            self.trackers[m[1]].update_with_box(dets[m[0]], frame)
        
        # create trackers for unmatched
        for i in unmatched_detections:
            trk = TrackerObject(dets[i], frame)
            self.trackers.append(trk)
        
        # remove old trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        # simply return all trackers
        return np.array([trk.get_bbox() for trk in self.trackers])
    
    def iou(self, bb_gt, bb_test):
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
            + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return o

    def linear_assignment(self, cost_matrix):
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

    def associate_detections_to_trackers(self, detections, trackers):
        """
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        iou_matrix = self.iou(detections, trackers)
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = self.linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))
        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
