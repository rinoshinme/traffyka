"""
yolov3 detector using opencv with darknet weights
"""

import cv2
import numpy as np
import time


class Yolov3Detector(object):
    def __init__(self, model_cfg, model_weights):
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.conf_threshold = 0.1
        self.nms_threshold = 0.6

        # print(self.get_output_names())
    
    def get_output_names(self):
        # layer_names = self.net.getLayerNames()
        # return [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return self.net.getUnconnectedOutLayersNames()
    
    def postprocess(self, outputs, image_width, image_height):
        class_ids = []
        confidences = []
        boxes = []
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # numbers are [center_x, center_y, width, height]
                if confidence > 0.5:
                    center_x = int(detection[0] * image_width)
                    center_y = int(detection[1] * image_height)
                    width = int(detection[2] * image_width)
                    height = int(detection[3] * image_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        output_boxes = []
        output_scores = []
        output_labels = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            score = confidences[i]
            label = class_ids[i]
            output_boxes.append([left, top, left + width, top + height])
            output_scores.append(score)
            output_labels.append(label)
        return output_boxes, output_scores, output_labels
    
    def detect(self, image):
        image_height = image.shape[0]
        image_width = image.shape[1]
        input_blob = cv2.dnn.blobFromImage(image, 1.0/255, (416, 416), None, True, False)
        self.net.setInput(input_blob)
        # run inference
        start_time = time.time()
        outputs = self.net.forward(self.get_output_names())
        end_time = time.time()
        print('yolov3 detection time: {:.06f}s'.format(end_time - start_time))
        return self.postprocess(outputs, image_width, image_height)
        
    def detect_image_file(self, image_path):
        image = cv2.imread(image_path)
        boxes, scores, labels = self.detect(image)
        drawn = self.draw(image, boxes, scores, labels)
        cv2.imshow('display', drawn)
        cv2.waitKey(0)

    def draw(self, image, boxes, scores, labels):
        drawn_image = image.copy()
        for box, score, label in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(drawn_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            text = '{}:{:.03f}'.format(label, score)
            cv2.putText(drawn_image, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return drawn_image
    
    def filter_labels(self, boxes, scores, labels, target_labels):
        if isinstance(target_labels, int):
            target_labels = [target_labels]
        out_boxes = []
        out_scores = []
        out_labels = []
        for box, score, label in zip(boxes, scores, labels):
            if label in target_labels:
                out_boxes.append(box)
                out_scores.append(score)
                out_labels.append(label)
        return out_boxes, out_scores, out_labels


if __name__ == '__main__':
    cfg = '../../model_data/yolov3.cfg'
    weights = '../../model_data/yolov3.weights'
    detector = Yolov3Detector(cfg, weights)
    image_path = '../../samples/sample1.jpg'
    detector.detect_image_file(image_path)
