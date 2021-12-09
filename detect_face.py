# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy
import os
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    # coords[:, 8].clamp_(0, img0_shape[1])  # x5
    # coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xywh, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(4):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], 5, )

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img



def detect_one(model, image_path, device, img_size=224, save_dir="./"):
    # Load model
    if not os.path.isdir(save_dir) and not os.path.isfile(save_dir):
        os.mkdir(save_dir)
    conf_thres = 0.35
    iou_thres = 0.2

    orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz, color=(114, 114, 114))[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]
    # print("time=== ", time_synchronized() - t1)
    # print(pred)

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    max_conf = 0
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()

            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                if max_conf < conf:
                    max_conf = conf
                landmarks = (det[j, 5:13].view(1, 8) / gn_lks).view(-1).tolist()
                class_num = det[j, 13].cpu().numpy()
                orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
    if max_conf < 1.5:
        print(os.path.join(save_dir, os.path.basename(image_path)))
        print(max_conf)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(image_path)), orgimg)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--image', type=str, default='data/images/test.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save-dir', type=str, default='./', help='source')  # file/folder, 0 for webcam

    parser.add_argument('--img-size', type=int, default=256, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)
    check_paths = '''
        2e4ada20f36cdc1ff81322514ac0bb09.jpeg
        3a237aac1b091230c1f4b13e2e28169e.jpeg
        9bac165f45d286c8ba86f2f0a4db1a07.jpeg
        abeef5d27c08ffa811983eb4a0444b3e.jpeg
        f9f0be60fdc2e4d4bcd6315526e10bca.jpeg
        2025728_back.jpg
        2020-07-03-18-58-27_001087007782_KIEU-MINH-TUAN_True.jpg
        2020-01-11-00-41-04_079195007107_NGUYEN-TRAN-KIM-SANG_True.jpg
        2020-01-18-19-01-17_250300919_NGUYEN-VAN-DOAN_False.jpg
        2020-01-19-15-36-18_079088013507_HUYNH-THANH-QUAN_False.jpg
        2020-01-19-16-47-59_211884267_NGUYEN-DO-THE-CUONG_True.jpg
        2020-01-24-16-34-26_001092007110_TRAN-NGOC-QUYNH_True.jpg
        2020-03-03-15-03-34_025176000075_NGUYEN-THI-HANG_True.jpg
        2020-02-28-13-00-29_001097010902_NGUYEN-BAO-HUNG_True.jpg
        2020-02-27-09-42-00_023853816_MAI-THIEN-PHUC_True.jpg
        2020-02-26-12-30-15_001090032297_NGUYEN-QUANG-DUNG_True.jpg
        2020-02-26-09-48-05_085068850_HA-THI-THU-PHUONG_True.jpg
        2020-02-19-15-00-03_132235504_HAN-ANH-TU_True.jpg
        2020-02-19-09-03-10_079186000041_TRUONG-CONG-SU_False.jpg
        2020-02-13-10-00-35_071074576_TRAN-QUY_False.jpg
        2020-02-12-09-22-41_341335455_TRUONG-THI-NGOC-MA_False.jpg
        2020-02-11-07-13-47_035094001873_DINH-VAN-MANH_True.jpg
        2020-02-09-09-36-33_187587754_NGUYEN-VAN-CONG_True.jpg
        2020-02-03-14-42-12_013370746_DAO-THI-HOA_False.jpg
        2020-01-28-19-14-44_145421605_TRAN-XUAN-DUONG_True.jpg
        2020-03-03-22-59-20_001092015308_PHUNG-VAN-MINH_#.jpg
        2020-03-13-09-15-59_291033487_NGUYEN-THI-HOAP-THUONG_False.jpg
        2020-03-14-14-34-44_205089291_TRAN-PHUOC-VINH_False.jpg
        2020-04-09-11-56-38_211031286_THAN_False.jpg
        2020-04-17-21-33-11_013370788_NGO-THI-TINH_False.jpg
        2020-04-30-12-02-26_079098009000_TRAN-MINH-TAN-THANH_#.jpg
        2020-05-23-17-36-21_197462165_MAI-LE-HAI-THANG-THIEU-SON_False.jpg
        2020-05-30-22-37-13_206075482_VO-THI-HOA-THANH-DU-THANH_False.jpg
        2020-06-04-11-59-35_001179003133_VU-THI-MINH-NGHI_False.jpg
        2020-06-04-14-10-10_022192002116_NGUYEN-KIM-CUC_True.jpg
        cavet_202104121501_engine=1S9A033381&chassis=S9A0CY033382_606d6009efffa285a37a910a_1618214473194_blob.jpg
        5b1f39e7d91f03183af1869351dbfa74.jpeg
    '''
    check_paths = [check_path.strip() for check_path in check_paths.strip().split("\n")]
    print(check_paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    if os.path.isdir(opt.image):
        for img in os.listdir(opt.image):
            if os.path.isdir(os.path.join(opt.image, img)):
                for im in os.listdir(os.path.join(opt.image, img)):
                    if im not in check_paths:
                        continue
                    if im.endswith(".jpg") or im.endswith(".jpeg") or im.endswith(".png") or im.endswith(".JPEG"):
                        detect_one(model, os.path.join(opt.image, img, im), device, opt.img_size, opt.save_dir)
            if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png") or img.endswith(".JPEG"):
                # if img not in check_paths:
                #     continue
                detect_one(model, os.path.join(opt.image, img), device, opt.img_size, opt.save_dir)
    else:
        detect_one(model, opt.image, device, opt.img_size, opt.save_dir)

# CUDA_VISIBLE_DEVICES="" python detect_face.py --weights runs/train/exp265/weights/71.pt --image /home/quannd/CardDetection/KeyPointDetection/yolov5-face/download3.png --img-size 256 --save-dir ../

