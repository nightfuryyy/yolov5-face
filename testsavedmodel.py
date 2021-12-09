# get input and output infor
# saved_model_cli show --dir /home/quannd/CardDetection/KeyPointDetection/yolov5-face/CardDetectionOnly_v1 --tag_set serve --signature_def serving_default

import numpy as np
from numpy.lib.type_check import imag
import cv2
import os

import tensorflow as tf
import time 
tf.config.experimental.enable_tensor_float_32_execution(False)

from save_model_ultils import anchor_process, show_results, letterbox, non_max_suppression_face, scale_coords, scale_coords_landmarks, xyxy2xywh, show_results_v2

class Model: 
    def __init__(self, model_path, anchors=[], output_names=[]):
        self.model_path =  model_path
        self.anchors = anchors  
        self.output_names = output_names
        self.init_model(model_path, anchors, output_names)

    def init_model(self, model_path, anchors, output_names):
        if len(anchors) == 0:
            # self.anchors = [[[23,29],  [43,55],  [73,105]], [[157,94],  [108,167],  [221,107]], [[155,190],  [236,156],  [241,218]]]
            # self.anchors = [[[80,50],  [60,90],  [120,70]], [[196,118],  [135,209],  [276,134]], [[194,238],  [295,195],  [301,272]]]
            self.anchors = [[[64,48], [64,64], [96,64]], [[161,98], [108,160], [226,105]], [[152,211],  [229,164],  [241,220]]]
        if len(output_names) == 0:
            self.output_names = ['output_0', 'output_1', 'output_2']
        self.model = tf.saved_model.load(model_path)
        self.model = self.model.signatures["serving_default"]
    def inference(self, image, image_size=320, conf_thres=0.5, iou_thres=0.2, return_time_infer=False):
        processed_image, img_for_visual = self.preprocess_scrath(image, image_size)
        tf_input = tf.convert_to_tensor(processed_image, dtype=tf.float32)
        t1 = time.time()
        outputs = self.model(tf_input)
        t2 = time.time()
        result = self.posprocess(outputs, img_shape=img_for_visual.shape[:2], orgimg_shape=image.shape, conf_thres=conf_thres, iou_thres=iou_thres) 
        # return result
        if return_time_infer:
            return self.convert_result(result, image.shape), t2 - t1 
        else:
            return self.convert_result(result, image.shape)
        

    def preprocess_scrath(self, orgi_image, img_size=320):
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
            gn_lks = np.asarray(orgimg_shape)[[1, 0, 1, 0, 1, 0, 1, 0]]  # normalization gain landmarks
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_shape, det[:, :4], orgimg_shape).round()
                if det.shape[1] > 6:
                    det[:, 5:13] = scale_coords_landmarks(img_shape, det[:, 5:13], orgimg_shape).round()
                res = []
                for j in range(det.shape[0]):
                    xywh = (xyxy2xywh(det[j, :4].reshape(1, 4)) / gn).reshape(-1).tolist()
                    conf = det[j, 4]
                    if det.shape[1] > 6:
                        landmarks = (det[j, 5:13].reshape(1, 8) / gn_lks).reshape(-1).tolist()
                        res.append([xywh, conf, landmarks])
                    else:
                        res.append([xywh, conf])                
                results.append(res)
            else: 
                results.append([])
        if len(results) == 1:
            results = results[0]
        return results

    def convert_result(self, result, img_shape):
        h, w, c = img_shape
        new_res = []
        for box in result:
            if len(box) > 2:
                xywh, conf, lm = box[0], box[1], box[2]
            else: 
                xywh, conf = box[0], box[1]
            xyxy = []
            xyxy.append(int((xywh[0] * w - 0.5 * xywh[2] * w)))
            xyxy.append(int((xywh[1] * h - 0.5 * xywh[3] * h)))
            xyxy.append(int((xywh[0] * w + 0.5 * xywh[2] * w)))
            xyxy.append(int((xywh[1] * h + 0.5 * xywh[3] * h)))
            if len(box) > 2:
                lm_unnor = []
                lm_unnor.append(int(lm[0] * w))
                lm_unnor.append(int(lm[1] * h))
                lm_unnor.append(int(lm[2] * w))
                lm_unnor.append(int(lm[3] * h))
                lm_unnor.append(int(lm[4] * w))
                lm_unnor.append(int(lm[5] * h))
                lm_unnor.append(int(lm[6] * w))
                lm_unnor.append(int(lm[7] * h))
                box = [xyxy, conf, lm_unnor]
            else: 
                box = [xyxy, conf]
            new_res.append(box)
        return list(sorted([[np.reshape(np.array(kps), (-1, 2)), bb, conf] for bb, conf, kps in new_res], \
                            key=lambda x: x[2], reverse=True))        


def main():
    # PB_MODEL_PATH = "/home/quannd/CardDetection/KeyPointDetection/yolov5-face/CardDetectionOnly_v1_dynamic_shape"
    # PB_MODEL_PATH = "/home/quannd/CardDetection/KeyPointDetection/yolov5-face/CardDetectionOnly_v1_dynamic_shape_best955"
    PB_MODEL_PATH = "CardDetectionOnly_v3_dynamic_shape_0.0079s_whiteborder_new_anchor_improve_cutted_object"
    PB_MODEL_PATH = "CardDetectionOnly_v3_dynamic_shape_0.0081_256_whiteborder_new_anchor_improve_cutted_object"
    PB_MODEL_PATH = "model/models/idcard_detection_v2/3"
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.2
    IMAGE_DIR = "../../../one_class_yolov5_dataset/val/"
    IMAGE_DIR = "sai_quan/"
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
        if not image_path.endswith(".jpg") and not image_path.endswith(".jpeg"):
            continue
    # image_paths = ["/home/quannd/CardDetection/KeyPointDetection/yolov5-face/download3.png", "/home/quannd/CardDetection/KeyPointDetection/yolov5-face/result.jpg", ]
    # for i in range(len(image_paths)):
        # image_path = image_paths[i]
        # print(image_path)
        image = cv2.imread(IMAGE_DIR + image_path)
        # image = cv2.imread(image_path)
        outputs = model.inference(image, IMAGE_SIZE, conf_thres=CONF_THRESHOLD, iou_thres=IOU_THRESHOLD)#, save_path=SAVE_DIR + image_path.split("/")[-1])
        save_path = SAVE_DIR + image_path.split("/")[-1]
        if len(outputs) != 0 and save_path != '':
            image = image.copy()
            for box in outputs:
                landmark, xywh, conf = box[0], box[1], box[2]
                print(landmark)
                image =  show_results_v2(image, xywh, conf, landmark.tolist())

                # if len(box) > 2:
                #     xywh, conf, landmark = box[0], box[1], box[2]
                #     image = show_results(image, xywh, conf, landmark)
                # else:
                #     xywh, conf = box[0], box[1]
                #     image = show_results(image, xywh, conf)
            cv2.imwrite(save_path, image)
        print("==============================")
        print(outputs)



if __name__ == '__main__':
    main()

# python detect_face.py --weights runs/train/exp261/weights/74.pt --image /home/quannd/CardDetection/KeyPointDetection/yolov5-face/result.jpg --img-size 256 --save-dir ../

