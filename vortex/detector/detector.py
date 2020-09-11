"""
Basic detector class, act as a unified detector interface.
"""
import cv2


class BaseDetector(object):
    def __init__(self):
        """
        load weights, config file, label file
        """
        pass

    def detect(self, image):
        """
        return boxes, scores and labels as numpy arrays
        """
        raise NotImplementedError

    def detect_image_file(self, image_path, visualize=False):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if not image:
            print('cannot open image file: {}'.format(image_path))
            return
        boxes, scores, labels = self.detect(image)
        if visualize:
            self.visualize(image, boxes, scores, labels)
        return boxes, scores, labels

    def detect_video_file(self, video_path):
        raise NotImplementedError

    def visualize(self, image, boxes, scores, labels):
        raise NotImplementedError
