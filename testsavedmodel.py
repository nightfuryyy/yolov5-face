# get input and output infor
# saved_model_cli show --dir /home/quannd/CardDetection/KeyPointDetection/yolov5-face/CardDetectionOnly_v1 --tag_set serve --signature_def serving_default

import numpy as np
from numpy.lib.type_check import imag
import cv2
import os

import tensorflow as tf
import time 
tf.config.experimental.enable_tensor_float_32_execution(False)

from save_model_ultils import anchor_process, show_results, letterbox, non_max_suppression_face, scale_coords, xyxy2xywh

class Model: 
    def __init__(self, model_path, anchors=[], output_names=[]):
        self.model_path =  model_path
        self.anchors = anchors  
        self.output_names = output_names
        self.init_model(model_path, anchors, output_names)

    def init_model(self, model_path, anchors, output_names):
        if len(anchors) == 0:
            self.anchors = [[[23,29],  [43,55],  [73,105]], [[157,94],  [108,167],  [221,107]], [[155,190],  [236,156],  [241,218]]]
        if len(output_names) == 0:
            self.output_names = ['output_0', 'output_1', 'output_2']
        self.model = tf.saved_model.load(model_path)
        self.model = self.model.signatures["serving_default"]
    def inference(self, image, image_size=256, conf_thres=0.5, iou_thres=0.2, save_path='', show_time_infer=False):
        processed_image, img_for_visual = self.preprocess_scrath(image, image_size)
        tf_input = tf.convert_to_tensor(processed_image, dtype=tf.float32)
        t1 = time.time()
        outputs = self.model(tf_input)
        t2 = time.time()
        if show_time_infer:
            print("TIME INFERENCE: ", t2 - t1)
        result = self.posprocess(outputs, img_shape=img_for_visual.shape[:2], orgimg_shape=image.shape, conf_thres=conf_thres, iou_thres=iou_thres)   
        if len(result) != 0 and save_path != '':
            image = image.copy()
            for box in result:
                xywh, conf = box[0], box[1]
                image = show_results(image, xywh, conf)
            cv2.imwrite(save_path, image)
        return result
        

    def preprocess_scrath(self, orgi_image, img_size=256):
        image = orgi_image.copy()
        h0, w0 = image.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            image = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=interp)
        image = letterbox(image, new_shape=img_size, auto=True, scaleFill=True)[0]
        img_after_letterbox = image.copy()
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = image.astype(np.float32) # uint8 to fp16/32
        image /= 255.0
        image = np.array([image])
        return image, img_after_letterbox


    def posprocess(self, outputs, img_shape, orgimg_shape, conf_thres=0.5, iou_thres=0.2):
        outputs = [outputs[output_name].numpy() for output_name in self.output_names]
        outputs = anchor_process(outputs, self.anchors)
        pred = non_max_suppression_face(outputs, conf_thres, iou_thres)
        # Process detections
        results = []
        for i, det in enumerate(pred):  # detections per image
            gn = np.asarray(orgimg_shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # gn_lks = np.asarray(orgimg_shape)[[1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain landmarks
            if len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = clip_coords(det[:, :4], img_shape)
                det[:, :4] = scale_coords(img_shape, det[:, :4], orgimg_shape).round()
                # det[:, 5:13] = scale_coords_landmarks(img_shape[2:], det[:, 5:13], orgimg.shape).round()
                res = []
                for j in range(det.shape[0]):
                    xywh = (xyxy2xywh(det[j, :4].reshape(1, 4)) / gn).reshape(-1).tolist()
                    # xywh = (xyxy2xywh(det[j, :4].reshape(1, 4))).reshape(-1).tolist()
                    conf = det[j, 4]
                    res.append([xywh, conf])                
                    # landmarks = (det[j, 5:13].view(1, 8) / gn_lks).view(-1).tolist()
                    # class_num = det[j, 13].cpu().numpy()
                    # orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
                results.append(res)
            else: 
                results.append([])
        if len(results) == 1:
            results = results[0]
        return results



def main():
    PB_MODEL_PATH = "/home/quannd/CardDetection/KeyPointDetection/yolov5-face/CardDetectionOnly_v1_dynamic_shape"
    # PB_MODEL_PATH = "/home/quannd/CardDetection/KeyPointDetection/yolov5-face/CardDetectionOnly_v1_dynamic_shape_best955"
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    IMAGE_DIR = "../../one_class_yolov5_dataset/val/"
    SAVE_DIR = "infer_savemodel_results/"
    IMAGE_SIZE = 256
    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    #init
    model = Model(PB_MODEL_PATH)
    # return 
    res = []
    count = 0
    c = 0
    for image_path in os.listdir(IMAGE_DIR):
        if not image_path.endswith(".jpg"):
            continue
    # image_paths = ["/home/quannd/CardDetection/KeyPointDetection/yolov5-face/download3.png", "/home/quannd/CardDetection/KeyPointDetection/yolov5-face/result.jpg", ]
    # for i in range(len(image_paths)):
        # image_path = image_paths[i]
        # print(image_path)
        image = cv2.imread(IMAGE_DIR + image_path)
        # image = cv2.imread(image_path)
        outputs = model.inference(image, IMAGE_SIZE, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD, save_path=SAVE_DIR + image_path.split("/")[-1])
        print("==============================")
        print(outputs)



if __name__ == '__main__':
    main()

# python detect_face.py --weights runs/train/exp261/weights/74.pt --image /home/quannd/CardDetection/KeyPointDetection/yolov5-face/result.jpg --img-size 256 --save-dir ../