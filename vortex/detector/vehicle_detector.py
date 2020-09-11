import cv2
import numpy as np
import os
import time
import onnxruntime


class VehicleDetector(object):
    def __init__(self, onnx_path, label_file):
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_names, self.output_names = self.get_io_names(self.onnx_session)
        print('input names: ', self.input_names)
        print('output_names: ', self.output_names)
        # define MODEL PARAMETERS
        self.class_names = self.load_class_names(label_file)
        self.image_size = (608, 608)
        self.conf_thresh = 0.1
        self.iou_thresh = 0.6

    def detect(self, image):
        inputs, ratio, dw, dh = self.preprocess(image)
        scores_array, boxes_array = self.forward(inputs)
        boxes = self.decode_outputs(boxes_array, scores_array, ratio, dw, dh)
        return boxes

    def detect_image_file(self, image_path):
        """
        for demo only
        """
        img0 = cv2.imread(image_path)
        if img0 is None:
            return
        inputs, ratio, dw, dh = self.preprocess(img0)
        scores_array, boxes_array = self.forward(inputs)
        boxes = self.decode_outputs(boxes_array, scores_array, ratio, dw, dh)
        image = self.draw_boxes(img0, boxes)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        cv2.imwrite('image.jpg', image)
        return boxes
    
    def detect_video_file(self, video_path):
        """
        for demo only
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print('cannot open video file {}'.format(video_path))
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            inputs, ratio, dw, dh = self.preprocess(frame)
            scores_array, boxes_array = self.forward(inputs)
            boxes = self.decode_outputs(boxes_array, scores_array, ratio, dw, dh)
            image = self.draw_boxes(frame, boxes)
            cv2.imshow('surveillance', image)
            cv2.waitKey(1)

    def draw_boxes(self, image, boxes):
        for box in boxes:
            xmin, ymin, xmax, ymax, score, name = box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=1)
            cv2.putText(image, '{}:{:.02f}'.format(name, score), (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return image

    # --------------------------------------------------------------------------------
    def load_class_names(self, label_file):
        labels = []
        with open(label_file, 'r') as f:
            for line in f.readlines():
                labels.append(line.strip().lower())
        return labels

    def forward(self, inputs):
        input_feed = {'inputs': inputs}
        outputs = self.onnx_session.run(self.output_names, input_feed=input_feed)
        return outputs[0], outputs[1]
    
    def get_io_names(self, onnx_session):
        input_names = []
        for node in onnx_session.get_inputs():
            input_names.append(node.name)
        output_names = []
        for node in onnx_session.get_outputs():
            output_names.append(node.name)
        return input_names, output_names

    def letterbox(self, image, new_shape=(608, 608)):
        height = image.shape[0]
        width = image.shape[1]
        if width < height:
            new_h = 608
            new_w = int(608 * width / height)
            ratio = 608 / height
            dh = 0
            dw = (608 - new_w) // 2
        else:
            new_w = 608
            new_h = int(608 * height / width)
            ratio = 608 / width
            dw = 0
            dh = (608 - new_h) // 2
        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = dh, 608 - new_h - dh
        left, right = dw, 608 - new_w - dw
        color = (114, 114, 114)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, (ratio, ratio), (dw, dh)

    def preprocess(self, image):
        img, ratio, (dw, dh) = self.letterbox(image, new_shape=self.image_size)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = img.astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_tensor, axis=0)
        return img_tensor, ratio, dw, dh

    def decode_outputs(self, boxes, scores, ratio, dw, dh):
        boxes[:, 0] = boxes[:, 0] * 608
        boxes[:, 2] = boxes[:, 2] * 608
        boxes[:, 1] = boxes[:, 1] * 608
        boxes[:, 3] = boxes[:, 3] * 608

        inputs = np.concatenate([boxes, scores], axis=1)
        inputs = np.expand_dims(inputs, axis=0)
        outputs = self.non_max_suppression_numpy(inputs, conf_thres=self.conf_thresh, 
                                                 iou_thresh=self.iou_thresh, multi_label=True)
        
        outputs = outputs[0]
        nboxes = outputs.shape[0]
        out_boxes = []
        for i in range(nboxes):
            xmin = int((outputs[i, 0] - dw) / ratio[0])
            ymin = int((outputs[i, 1] - dh) / ratio[1])
            xmax = int((outputs[i, 2] - dw) / ratio[0])
            ymax = int((outputs[i, 3] - dh) / ratio[1])
            score = outputs[i, 4]
            label = int(outputs[i, 5])
            class_name = self.class_names[label]
            out_boxes.append([xmin, ymin, xmax, ymax, score, class_name])
        out_boxes = np.array(out_boxes)
        return out_boxes
    
    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    def py_cpu_nms(self, boxes, scores, thresh):
        """
        nms
        :param dets: ndarray [x1,y1,x2,y2,score]
        :param thresh: int
        :return: list[index]
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        order = scores.argsort()[::-1]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            over = (w * h) / (area[i] + area[order[1:]] - w * h)
            index = np.where(over <= thresh)[0]
            order = order[index + 1] # 不包括第0个
        return keep

    def non_max_suppression_numpy(self, prediction, conf_thres=0.1, iou_thresh=0.6, multi_label=False, classes=None, agnostic=False):
        # Settings
        merge = True  # merge for best mAP
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        time_limit = 10.0  # seconds to quit after

        t = time.time()

        nc = prediction[0].shape[1] - 5  # number of classes
        # multi_label &= nc > 1  # multiple labels per box
        multi_label = False
        output = [None] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            x = x[x[:, 4] > conf_thres]  # confidence
            x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero().t()
                x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
            else:  # best class only
                # conf, j = x[:, 5:].max(1)
                conf = x[:, 5:].max(1)
                j = x[:, 5:].argmax(1)
                # x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]
                conf1 = np.expand_dims(conf, axis=1)
                j = np.expand_dims(j.astype(np.float), axis=1)

                x = np.concatenate((box, conf1, j), 1)[conf > conf_thres]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue

            # Sort by confidence
            # x = x[x[:, 4].argsort(descending=True)]

            # Batched NMS
            c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
            boxes, scores = x[:, :4] + np.expand_dims(c, axis=1) * max_wh, x[:, 4]  # boxes (offset by class), scores
            # i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
            i = self.py_cpu_nms(boxes, scores, iou_thresh)

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break  # time limit exceeded
        return output


if __name__ == '__main__':
    onnx_path = 'model_data/yolo_vehicle.onnx'
    label_file = 'model_data/vehicle_v1/traffic.names'
    demo = VehicleDetector(onnx_path, label_file)
    video_path = 'D:/workspace/traffic/data/Relaxing highway traffic.mp4'
    demo.detect_video_file(video_path)
