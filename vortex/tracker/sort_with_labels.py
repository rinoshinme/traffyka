"""
Modified version of SORT multi-object tracking
"""
from collections import defaultdict
import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox, label):
        self.kf = self.build_kf(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        # self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        # self.label_history = []
        self.label_history = defaultdict(int)
        self.label_history[int(label)] += 1

    def build_kf(self, bbox):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  
                              [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        kf.R[2:,2:] *= 10.
        kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        kf.P *= 10.
        kf.Q[-1,-1] *= 0.01
        kf.Q[4:,4:] *= 0.01
        kf.x[:4] = self.convert_bbox_to_z(bbox)
        return kf

    def update(self, bbox, label):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        # self.label_history.append(label)
        self.label_history[int(label)] += 1

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_majority_label(self):
        max_count = 0
        max_label = None
        print(self.label_history)
        for k in self.label_history.keys():
            if self.label_history[k] > max_count:
                max_count = self.label_history[k]
                max_label = k
        return max_label, max_count

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    def convert_bbox_to_z(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    #scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if(score==None):
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    
    def update(self, dets, labels):
        """
        dets: [N, 5] with (xmin, ymin, xmax, ymax, score)
        labels: list of integer category indices.
        return: [N, 6] with (xmin, ymin, xmax, ymax, obj_index, obj_label)
        """
        self.frame_count += 1
        to_del = []
        trks = []
        for t, tracker in enumerate(self.trackers):
            pos = tracker.predict()[0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                continue
            trks.append([pos[0], pos[1], pos[2], pos[3], 0])
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)
        # update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]], labels[m[0]])
        # create new trackers
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], labels[i])
            self.trackers.append(trk)
        # update unmatched trackers (and delete if necessary)
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            max_label, max_count = trk.get_majority_label()
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1], [max_label])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))

    # helper functions
    def linear_assignment(self, cost_matrix):
        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            return np.array([[y[i], i] for i in x if i >= 0])
        except ImportError:
            from scipy.optimize import linear_sum_assignment
            x, y = linear_sum_assignment(cost_matrix)
            return np.array(list(zip(x, y)))

    def iou_batch(self, bb_test, bb_gt):
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

    def associate_detections_to_trackers(self, detections, trackers):
        """
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        iou_matrix = self.iou_batch(detections, trackers)
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
