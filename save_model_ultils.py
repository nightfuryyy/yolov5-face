import cv2 
import torch  
import numpy as np 
import time  
import torchvision
def sigmoid(z):
    z = np.clip(z, -88.72, 88.72)
    return 1/(1 + np.exp(-z))

def convert_result(result, img_shape):
    h, w, c = img_shape
    new_res = []
    if len(result) != 0: 
        for box in result:
            xywh, conf = box[0], box[1]
            xyxy = []
            xyxy.append(int((xywh[0] * w - 0.5 * xywh[2] * w)))
            xyxy.append(int((xywh[1] * h - 0.5 * xywh[3] * h)))
            xyxy.append(int((xywh[0] * w + 0.5 * xywh[2] * w)))
            xyxy.append(int((xywh[1] * h + 0.5 * xywh[3] * h)))
            box = [xyxy, conf]
            new_res.append(box)
    return new_res


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (np.min(box1[:, None, 2:], box2[:, 2:]) -
             np.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)

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
    coords[:, 0] = np.clip(coords[:, 0], 0, img0_shape[1])
    coords[:, 1] = np.clip(coords[:, 1], 0, img0_shape[0])
    coords[:, 2] = np.clip(coords[:, 2], 0, img0_shape[1])
    coords[:, 3] = np.clip(coords[:, 3], 0, img0_shape[0])
    coords[:, 4] = np.clip(coords[:, 4], 0, img0_shape[1])
    coords[:, 5] = np.clip(coords[:, 5], 0, img0_shape[0])
    coords[:, 6] = np.clip(coords[:, 6], 0, img0_shape[1])
    coords[:, 7] = np.clip(coords[:, 7], 0, img0_shape[0])
    return coords

def non_max_suppression_face(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 15  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    if prediction[0].shape[1] > 6:
        is_contain_landmark = True  
    else:
        is_contain_landmark = False
    
    
    t = time.time()
    if is_contain_landmark:
        output = [np.zeros((0, 14))] * prediction.shape[0]
    else: 
        output = [np.zeros((0, 6))] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        x[:, -1:] *= x[:, 4:5]

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        conf = np.max(x[:, -1:], axis=1, keepdims=True)
        j = np.argmax(x[:, -1:], axis=1).reshape(conf.shape)
        if is_contain_landmark:
            x = np.concatenate((box, conf, x[:, 5:13], j.astype(np.float32)), 1)[conf.reshape(-1) > conf_thres]
        else: 
            x = np.concatenate((box, conf, j.astype(np.float32)), 1)[conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.tensor(classes, device=x.device)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, -1:] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_thres).numpy()  # NMS
        #if i.shape[0] > max_det:  # limit detections
        #    i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = np.dot(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1] - 1)  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0] - 1)  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1] - 1)  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0] - 1)  # y2
    return boxes

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def show_results(img, xywh, conf, landmarks=[]):
    h, w, c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int((xywh[0] * w - 0.5 * xywh[2] * w))
    y1 = int((xywh[1] * h - 0.5 * xywh[3] * h))
    x2 = int((xywh[0] * w + 0.5 * xywh[2] * w))
    y2 = int((xywh[1] * h + 0.5 * xywh[3] * h))
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), int(tl), cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    if len(landmarks) != 0:
        lm = landmarks
        lm_unnor = []
        lm_unnor.append(int(lm[0] * w))
        lm_unnor.append(int(lm[1] * h))
        lm_unnor.append(int(lm[2] * w))
        lm_unnor.append(int(lm[3] * h))
        lm_unnor.append(int(lm[4] * w))
        lm_unnor.append(int(lm[5] * h))
        lm_unnor.append(int(lm[6] * w))
        lm_unnor.append(int(lm[7] * h))
        landmarks = lm_unnor
        for i in range(4):
            point_x = int(landmarks[2 * i])
            point_y = int(landmarks[2 * i + 1])
            cv2.circle(img, (point_x, point_y), tl+1, clors[i], 5, )

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        return img, img

    dw /= 2  # divide padding into 2 sides 
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().numpy()

def anchor_process(outputs, anchors):
#     anchors:
#   - [4,5,  8,10,  13,16]  # P3/8
#   - [23,29,  43,55,  73,105]  # P4/16
#   - [146,217,  231,300,  335,433] 
    new_outputs = []
    anchors = np.asarray(anchors).reshape(3, 1, -1, 1, 1, 2)
    for i, x in enumerate(outputs):
        # if grid.shape[2:4] != x.shape[2:4]:
        #     self.grid[i] = self._make_grid(nx, ny)
        output_shape = x.shape
        grid = make_grid(output_shape[-2], output_shape[-3])

        y = np.full_like(x, 0)
        # class_range = list(range(13)) + list(range(output_shape[-1] - 1, output_shape[-1]))
        # y[..., class_range] = sigmoid(x[..., class_range])
        # y[..., 5:13] = x[i][..., 5:13]
        y = sigmoid(x)
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * 2**(i + 3)  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchors[i]  # wh

        y[..., 5:13] = y[..., 5:13] * 8 - 4

        y[..., 5:7]   = y[..., 5:7] *   anchors[i] + grid * 2**(i + 3) # landmark x1 y1
        y[..., 7:9]   = y[..., 7:9] *   anchors[i] + grid * 2**(i + 3)# landmark x2 y2
        y[..., 9:11]  = y[..., 9:11] *  anchors[i] + grid * 2**(i + 3)# landmark x3 y3
        y[..., 11:13] = y[..., 11:13] * anchors[i] + grid * 2**(i + 3)# landmark x4 y4

        #y[..., 5:7] = (y[..., 5:7] * 2 -1) * self.anchor_grid[i]  # landmark x1 y1
        #y[..., 7:9] = (y[..., 7:9] * 2 -1) * self.anchor_grid[i]  # landmark x2 y2
        #y[..., 9:11] = (y[..., 9:11] * 2 -1) * self.anchor_grid[i]  # landmark x3 y3
        #y[..., 11:13] = (y[..., 11:13] * 2 -1) * self.anchor_grid[i]  # landmark x4 y4
        #y[..., 13:15] = (y[..., 13:15] * 2 -1) * self.anchor_grid[i]  # landmark x5 y5

        new_outputs.append(y.reshape(output_shape[0], -1, output_shape[-1]))
# return x if self.training else (torch.cat(z, 1), x)
    return np.concatenate(new_outputs, 1)
