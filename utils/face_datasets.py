import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
from shapely.geometry import Polygon
import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import xyxy2xywh, xywh2xyxy, clean_str
from utils.torch_utils import torch_distributed_zero_first


# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix=''):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadFaceImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                    )

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadFaceImagesAndLabels.collate_fn4 if quad else LoadFaceImagesAndLabels.collate_fn)
    return dataloader, dataset
class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class LoadFaceImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, rank=-1):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.augmentations = get_augmentations()

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                else:
                    raise Exception('%s does not exist' % p)
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            assert self.img_files, 'No images found'
        except Exception as e:
            raise Exception('Error loading data from %s: %s\nSee %s' % (path, e, help_url))

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files) or 'results' not in cache:  # changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Display cache
        [nf, nm, ne, nc, n] = cache.pop('results')  # found, missing, empty, corrupted, total
        desc = f"Scanning '{cache_path}' for images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        tqdm(None, desc=desc, total=n, initial=n)
        assert nf > 0 or not augment, f'No labels found in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        cache.pop('hash')  # remove hash
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def cache_labels(self, path=Path('./labels.cache')):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                    if len(l):
                        # assert l.shape[1] == 15, 'labels require 15 columns each'
                        assert (l >= -1).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 13), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 13), dtype=np.float32)
                x[im_file] = [l, shape]
            except Exception as e:
                nc += 1
                print('WARNING: Ignoring corrupted image and/or label %s: %s' % (im_file, e))

            pbar.desc = f"Scanning '{path.parent / path.stem}' for images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        if nf == 0:
            print(f'WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = [nf, nm, ne, nc, i + 1]
        torch.save(x, path)  # save for next time
        logging.info(f"New cache created: {path}")
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic_face(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic_face(self, random.randint(0, self.n - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            # img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            # shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            #resize only #scaleFill = True
            img, ratio, pad = letterbox(img, shape, auto=False, scaleFill=True, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad) 
            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

                #labels[:, 5] = ratio[0] * w * x[:, 5] + pad[0]  # pad width
                labels[:, 5] = np.array(x[:, 5] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 5] + pad[0]) + (
                    np.array(x[:, 5] >= 0, dtype=np.int32) - 1)
                labels[:, 6] = np.array(x[:, 6] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 6] + pad[1]) + (
                    np.array(x[:, 6] >= 0, dtype=np.int32) - 1)
                labels[:, 7] = np.array(x[:, 7] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 7] + pad[0]) + (
                    np.array(x[:, 7] >= 0, dtype=np.int32) - 1)
                labels[:, 8] = np.array(x[:, 8] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 8] + pad[1]) + (
                    np.array(x[:, 8] >= 0, dtype=np.int32) - 1)
                labels[:, 9] = np.array(x[:, 5] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 9] + pad[0]) + (
                    np.array(x[:, 9] >= 0, dtype=np.int32) - 1)
                labels[:, 10] = np.array(x[:, 5] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 10] + pad[1]) + (
                    np.array(x[:, 10] >= 0, dtype=np.int32) - 1)
                labels[:, 11] = np.array(x[:, 11] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 11] + pad[0]) + (
                    np.array(x[:, 11] >= 0, dtype=np.int32) - 1)
                labels[:, 12] = np.array(x[:, 12] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 12] + pad[1]) + (
                    np.array(x[:, 12] >= 0, dtype=np.int32) - 1)
                # labels[:, 13] = np.array(x[:, 13] > 0, dtype=np.int32) * (ratio[0] * w * x[:, 13] + pad[0]) + (
                #     np.array(x[:, 13] > 0, dtype=np.int32) - 1)
                # labels[:, 14] = np.array(x[:, 14] > 0, dtype=np.int32) * (ratio[1] * h * x[:, 14] + pad[1]) + (
                #     np.array(x[:, 14] > 0, dtype=np.int32) - 1)

        if self.augment:
            # Augment imagespace
            if not mosaic:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])
            # Augment colorspace
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

            labels[:, [5, 7, 9, 11]] /= img.shape[1]  # normalized landmark x 0-1
            labels[:, [5, 7, 9, 11]] = np.where(labels[:, [5, 7, 9, 11]] < 0, -1, labels[:, [5, 7, 9, 11]])
            labels[:, [6, 8, 10, 12]] /= img.shape[0]  # normalized landmark y 0-1
            labels[:, [6, 8, 10, 12]] = np.where(labels[:, [6, 8, 10, 12]] < 0, -1, labels[:, [6, 8, 10, 12]])

        # print(labels)
        if self.augment:
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

                    labels[:, [6, 8, 10, 12]] = 1 - labels[:, [6, 8, 10, 12]]
                    
                    # labels[:, 6] = np.where(labels[:,6] < 0, -1, 1 - labels[:, 6])
                    # labels[:, 8] = np.where(labels[:, 8] < 0, -1, 1 - labels[:, 8])
                    # labels[:, 10] = np.where(labels[:, 10] < 0, -1, 1 - labels[:, 10])
                    # labels[:, 12] = np.where(labels[:, 12] < 0, -1, 1 - labels[:, 12])

                    # labels[:, 14] = np.where(labels[:, 14] < 0, -1, 1 - labels[:, 14])

            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

                    labels[:, [5, 7, 9, 11]] = 1 - labels[:, [5, 7, 9, 11]]

                    # labels[:, 5] = np.where(labels[:, 5] < 0, -1, 1 - labels[:, 5])
                    # labels[:, 7] = np.where(labels[:, 7] < 0, -1, 1 - labels[:, 7])
                    # labels[:, 9] = np.where(labels[:, 9] < 0, -1, 1 - labels[:, 9])
                    # labels[:, 11] = np.where(labels[:, 11] < 0, -1, 1 - labels[:, 11])

                    # labels[:, 13] = np.where(labels[:, 13] < 0, -1, 1 - labels[:, 13])

        labels_out = torch.zeros((nL, 14))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
            # showlabels(img, labels[:, 1:5], labels[:, 5:13], self.img_files[index][:-4] + str(random.randint(0, 1000)) + ".jpg")

        if self.augment:
            img = self.augmentations(image=img)["image"] 

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        # print(index, '   --- labels_out: ', labels_out)
        # if nL:
        #     print( ' : landmarks : ', torch.max(labels_out[:, 5:15]), '  ---   ', torch.min(labels_out[:, 5:15]))
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

# back_2052353 # back_2031421 # 1615946796202
import albumentations
def get_augmentations():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_augmentations = albumentations.Compose(
        [
            # albumentations.Transpose(p=0.5),
            # albumentations.HorizontalFlip(p=0.5),
            # albumentations.VerticalFlip(p=0.5),
            # albumentations.ShiftScaleRotate(shift_limit=0.0, 
            #     scale_limit=0.0, rotate_limit=5, interpolation=1, border_mode=4, value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, always_apply=False, p=0.3
            # ),
            # albumentations.OneOf([
            # albumentations.augmentations.crops.transforms.RandomSizedCrop([int(cfg.data.input_size[0] * 0.95), cfg.data.input_size[0]], cfg.data.input_size[0], cfg.data.input_size[1], w2h_ratio=0.75, interpolation=1, always_apply=False, p=1.0),
            # albumentations.Sequential([
            #     albumentations.augmentations.geometric.transforms.Affine(
            #         # scale=(0.95, 1.0), 
            #         rotate=5, 
            #         fit_output=True,
            #         p=1.0
            #     ),
            #     albumentations.augmentations.geometric.resize.Resize(cfg.data.input_size[0], cfg.data.input_size[1])
            #     ])
            # ], p=0.4),
            
            # albumentations.augmentations.transforms.ColorJitter(
            #     brightness=0.1, contrast=0.3, saturation=0.3, hue=0.015, p = 0.5
            #     # brightness=0.3, contrast=0.3, saturation=0.7, hue=0.0, p = 0.5
            # ),
            # albumentations.augmentations.transforms.Equalize(
            #     by_channels=False,
            #     p=0.2
            # ),
            albumentations.OneOf([
                albumentations.Blur(
                    blur_limit=(3, 11),
                    p=0.8
                ),
                albumentations.MedianBlur(
                    blur_limit=(3,5),
                    p=0.1
                ),
                albumentations.MotionBlur(
                    blur_limit=(3,5),
                    p=0.1
                ),
            ], p=0.3),
            # albumentations.augmentations.transforms.CLAHE(
            #     clip_limit=8.0, tile_grid_size=(8, 8), always_apply=False, p=0.3
            # ),
            # albumentations.augmentations.transforms.GaussNoise(
            #     var_limit=(0, 20), p=0.2
            # ),
            # albumentations.augmentations.transforms.JpegCompression(
            #     quality_lower=99, quality_upper=100, p=1.0
            # ),
        ]
    )

    return train_augmentations

def showlabels(img, boxs, landmarks, path):
    img = img.copy()
    for box in boxs:
        x,y,w,h = box[0] * img.shape[1], box[1] * img.shape[0], box[2] * img.shape[1], box[3] * img.shape[0]
        #cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.rectangle(img, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)

    for landmark in landmarks:
        #cv2.circle(img,(60,60),30,(0,0,255))
        for i in range(4):
            if landmark[2*i] >= 1.0 or landmark[2*i] < 0 or landmark[2*i+1] >= 1.0 or landmark[2*i+1] < 0:
                print("aaaaaaaaaaaaaaaaaa")
                # nho thiet ket lai gen label
                print(path)
                print(landmark)
            cv2.circle(img, (int(landmark[2*i] * img.shape[1]), int(landmark[2*i+1]*img.shape[0])), 3 ,(0,0,255), -1)
    path = path.replace("/", "_")
    # print(path)
    cv2.imwrite("test_image/" + path + str(random.randint(0, 1000)) + "test.jpg", img)
    # cv2.imshow('test', img)
    # cv2.waitKey(0)

def find_intersection(e1, e2) :
    p0, p1 = e1
    p2, p3 = e2
    s10_x = p1[0] - p0[0]
    s10_y = p1[1] - p0[1]
    s32_x = p3[0] - p2[0]
    s32_y = p3[1] - p2[1]

    denom = s10_x * s32_y - s32_x * s10_y

    if denom == 0 : return None # collinear

    denom_is_positive = denom > 0

    s02_x = p0[0] - p2[0]
    s02_y = p0[1] - p2[1]

    s_numer = s10_x * s02_y - s10_y * s02_x

    if (s_numer < 0) == denom_is_positive : return None # no collision

    t_numer = s32_x * s02_y - s32_y * s02_x

    if (t_numer < 0) == denom_is_positive : return None # no collision

    if (s_numer > denom) == denom_is_positive or (t_numer > denom) == denom_is_positive : return None # no collision


    # collision detected

    t = t_numer / denom

    intersection_point = [ p0[0] + (t * s10_x), p0[1] + (t * s10_y) ]


    return intersection_point

def correct_landmark_one_point(landmark, num_wrong, four_edges):
    num_wrong = num_wrong * 2

    wrong_point = [landmark[num_wrong], landmark[num_wrong + 1]]
    next_point = [landmark[(num_wrong + 2) % 8], landmark[(num_wrong + 3) % 8]]
    previous_point = [landmark[(num_wrong + 6) % 8], landmark[(num_wrong + 7) % 8]]
    for edge in four_edges:
        col_pre = find_intersection(edge, [previous_point, wrong_point])
        if col_pre is not None:
            break
    for edge in four_edges:
        col_next = find_intersection(edge, [wrong_point, next_point])
        if col_next is not None:
            break 
    
    if col_pre is not None:
        landmark_pre = landmark.copy()
        landmark_pre[num_wrong], landmark[num_wrong + 1] = col_pre
        landmark_pre_poly = [(landmark_pre[i * 2], landmark_pre[i * 2 + 1]) for i in range(4)]
        pgon_pre = Polygon(landmark_pre_poly).area 
    else:
        pgon_pre = -1
        
    if col_next is not None:
        landmark_nx = landmark.copy()
        landmark_nx[num_wrong], landmark[num_wrong + 1] = col_next
        landmark_nx_poly = [(landmark_nx[i * 2], landmark_nx[i * 2 + 1]) for i in range(4)]
        pgon_nx = Polygon(landmark_nx_poly).area 
    else:
        pgon_nx = -1

    if pgon_nx == -1 and pgon_pre == -1:
        print(landmark)
        print(wrong_point)
    else:
        landmark = landmark_nx if pgon_nx > pgon_pre else landmark_pre
    return landmark

INT_MAX = 10000
 
# Given three collinear points p, q, r, 
# the function checks if point q lies
# on line segment 'pr'
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
     
    if ((q[0] <= max(p[0], r[0])) &
        (q[0] >= min(p[0], r[0])) &
        (q[1] <= max(p[1], r[1])) &
        (q[1] >= min(p[1], r[1]))):
        return True
         
    return False
 
# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p:tuple, q:tuple, r:tuple) -> int:
     
    val = (((q[1] - p[1]) *
            (r[0] - q[0])) -
           ((q[0] - p[0]) *
            (r[1] - q[1])))
            
    if val == 0:
        return 0
    if val > 0:
        return 1 # Collinear
    else:
        return 2 # Clock or counterclock
 
def doIntersect(p1, q1, p2, q2):
     
    # Find the four orientations needed for 
    # general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
 
    # General case
    if (o1 != o2) and (o3 != o4):
        return True
     
    # Special Cases
    # p1, q1 and p2 are collinear and
    # p2 lies on segment p1q1
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True
 
    # p1, q1 and p2 are collinear and
    # q2 lies on segment p1q1
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True
 
    # p2, q2 and p1 are collinear and
    # p1 lies on segment p2q2
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True
 
    # p2, q2 and q1 are collinear and
    # q1 lies on segment p2q2
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True
 
    return False
 
# Returns true if the point p lies 
# inside the polygon[] with n vertices
def isInside(points:list, p:tuple) -> bool:
    points = [point[0] for point in points]
    n = len(points)
     
    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False
         
    # Create a point for line segment
    # from p to infinite
    extreme = (INT_MAX, p[1])
    count = i = 0
     
    while True:
        next = (i + 1) % n
         
        # Check if the line segment from 'p' to 
        # 'extreme' intersects with the line 
        # segment from 'polygon[i]' to 'polygon[next]'
        if (doIntersect(points[i],
                        points[next],
                        p, extreme)):
                             
            # If the point 'p' is collinear with line 
            # segment 'i-next', then check if it lies 
            # on segment. If it lies, return true, otherwise false
            if orientation(points[i], p,
                           points[next]) == 0:
                return onSegment(points[i], p,
                                 points[next])
                                  
            count += 1
             
        i = next
         
        if (i == 0):
            break
         
    # Return true if count is odd, false otherwise
    return (count % 2 == 1)


def correct_landmark(landmarks, width, height, four_edges=[]):
    # neu co tren 2 diem nam ngoai anh => tra ve rong 
    # neu ko co diem nao sai => tra ve nhu cu 
    # neu co diem nam ngoai, vi du A, 2 diem lien ke la B C, tim giao giua AB va canh anh, va AC va canh anh goi la 
    # E F, neu thay the A thanh E hay F ma hinh tu giac co dien tich lon nhat thi ta chon diem do lam diem thay the 
    # roi tra ve landmark da dc thay the
    new_landmarks = []
    four_edges = [[[0, 0], [width, 0]], [[width, 0], [width, height]], [[width, height], [0, height]], [[0, height], [0, 0]]]
    # print(four_edges)
    for landmark in landmarks:
        landmark = landmark.tolist()
        num_wrong = []
        for i in range(4):
            if not isInside(four_edges, [landmark[2 * i], landmark[2 * i + 1]]):
                num_wrong.append(i)

        # print(len(num_wrong))
        # print(four_edges)
        # print(landmark)
        # print(num_wrong)
        if len(num_wrong) == 0:
            new_landmarks.append(landmark)
            continue 

        if len(num_wrong) == 1:
            new_landmarks.append(correct_landmark_one_point(landmark, num_wrong[0], four_edges))

        if len(num_wrong) == 2:
            tmp_landmark = correct_landmark_one_point(landmark, num_wrong[0], four_edges)
            new_landmarks.append(correct_landmark_one_point(tmp_landmark, num_wrong[1], four_edges))
        
        if len(num_wrong) >= 3:
            new_landmarks.append(landmark)
        # if len(num_wrong) == 3:
        #     for i in range(4):
        #         start = -1 
        #         for num in num_wrong:
        #             if i == num:
        #                 start = i
        #                 break
        #         if start == -1:
        #             start = i + 1
        #             break
        #     for i in range(3):
        #         tmp_landmark = landmark
        #         tmp_landmark = correct_landmark_one_point(tmp_landmark, (start + i) % 4, four_edges)
        #     new_landmarks.append(tmp_landmark)

        # if len(num_wrong) == 4:
        #     new_landmarks.append([-1 for i in range(8)]) # let it auto correct

    return new_landmarks

def load_mosaic_face(self, index):
    # loads images in a mosaic
    labels4 = []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + [self.indices[random.randint(0, self.n - 1)] for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            #box, x1,y1,x2,y2
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            #10 landmarks

            labels[:, 5] = np.array(x[:, 5] > 0, dtype=np.int32) * (w * x[:, 5] + padw) + (np.array(x[:, 5] > 0, dtype=np.int32) - 1)
            labels[:, 6] = np.array(x[:, 6] > 0, dtype=np.int32) * (h * x[:, 6] + padh) + (np.array(x[:, 6] > 0, dtype=np.int32) - 1)
            labels[:, 7] = np.array(x[:, 7] > 0, dtype=np.int32) * (w * x[:, 7] + padw) + (np.array(x[:, 7] > 0, dtype=np.int32) - 1)
            labels[:, 8] = np.array(x[:, 8] > 0, dtype=np.int32) * (h * x[:, 8] + padh) + (np.array(x[:, 8] > 0, dtype=np.int32) - 1)
            labels[:, 9] = np.array(x[:, 9] > 0, dtype=np.int32) * (w * x[:, 9] + padw) + (np.array(x[:, 9] > 0, dtype=np.int32) - 1)
            labels[:, 10] = np.array(x[:, 10] > 0, dtype=np.int32) * (h * x[:, 10] + padh) + (np.array(x[:, 10] > 0, dtype=np.int32) - 1)
            labels[:, 11] = np.array(x[:, 11] > 0, dtype=np.int32) * (w * x[:, 11] + padw) + (np.array(x[:, 11] > 0, dtype=np.int32) - 1)
            labels[:, 12] = np.array(x[:, 12] > 0, dtype=np.int32) * (h * x[:, 12] + padh) + (np.array(x[:, 12] > 0, dtype=np.int32) - 1)
            # labels[:, 13] = np.array(x[:, 13] > 0, dtype=np.int32) * (w * x[:, 13] + padw) + (np.array(x[:, 13] > 0, dtype=np.int32) - 1)
            # labels[:, 14] = np.array(x[:, 14] > 0, dtype=np.int32) * (h * x[:, 14] + padh) + (np.array(x[:, 14] > 0, dtype=np.int32) - 1)
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:5], 0, 2 * s, out=labels4[:, 1:5])  # use with random_perspective
        # img4, labels4 = replicate(img4, labels4)  # replicate

        #landmarks
        labels4[:, 5:] = np.where(labels4[:, 5:] < 0, -1, labels4[:, 5:])
        labels4[:, 5:] = np.where(labels4[:, 5:] > 2 * s, -1, labels4[:, 5:])

        labels4[:, 5] = np.where(labels4[:, 6] == -1, -1, labels4[:, 5])
        labels4[:, 6] = np.where(labels4[:, 5] == -1, -1, labels4[:, 6])

        labels4[:, 7] = np.where(labels4[:, 8] == -1, -1, labels4[:, 7])
        labels4[:, 8] = np.where(labels4[:, 7] == -1, -1, labels4[:, 8])

        labels4[:, 9] = np.where(labels4[:, 10] == -1, -1, labels4[:, 9])
        labels4[:, 10] = np.where(labels4[:, 9] == -1, -1, labels4[:, 10])

        labels4[:, 11] = np.where(labels4[:, 12] == -1, -1, labels4[:, 11])
        labels4[:, 12] = np.where(labels4[:, 11] == -1, -1, labels4[:, 12])

        # labels4[:, 13] = np.where(labels4[:, 14] == -1, -1, labels4[:, 13])
        # labels4[:, 14] = np.where(labels4[:, 13] == -1, -1, labels4[:, 14])

    # Augment
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove
    return img4, labels4


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])

def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


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

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def get_box_from_landmark(landmarks, n, height, width):
    x = landmarks[:, ::2] 
    y = landmarks[:, 1::2]
    xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
    xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)    
    xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
    return xy

# so diem tra ve bi sai khi aug dan toi bi mat diem

def save_poly(img, img_four_points):
    img = img.copy()
    isClosed = True
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 2
    pts = np.asarray(img_four_points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    # Using cv2.polylines() method
    # Draw a Blue polygon with 
    # thickness of 1 px
    img = cv2.polylines(img, [pts], 
                        isClosed, color, thickness)
    cv2.imwrite("test.jpg", img)

def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    if random.random() < 0.2:
        a += random.choice([90])  # add 90deg rotations to small rotations
    if random.random() < 0.5:
        s = random.uniform(1 - scale, 1)
    else:
        s = 1
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    n = len(targets)
    if n:
        T_tmp = np.eye(3)
        T_tmp[0, 2] = 0.5 * width  # x translation (pixels)
        T_tmp[1, 2] = 0.5 * height
        M_wo_T = T_tmp @ S @ R @ P @ C
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [5, 6, 7, 8, 9, 10, 11, 12]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M_wo_T.T  #   transform
        # print(xy[:, 2:3])
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 4, 2)  # rescale
        else:
            xy = xy[:, :2].reshape(n, 4, 2)
        # print("checkkkkkkkkkkk")
        # print(xy)
        x_min = max(-xy[:, :, 0].min(), 0)
        y_min = max(-xy[:, :, 1].min(), 0) 
        x_max = max(xy[:, :, 0].max() - width + 1, 0)
        y_max = max(xy[:, :, 1].max() - height + 1, 0) 
        # print(x_min, y_min, x_max, y_max) 
        # T = np.eye(3)
        # T[0, 2] = int(0.5 * width + 0.5 * (x_max - x_min))  # x translation (pixels)
        # T[1, 2] = int(0.5 * height + 0.5 * (y_max - y_min)) 
        # new_width = int(width + (x_max + x_min)+ 0.0 * random.random() * width)
        # new_height = int(height + (y_max + y_min) + 0.0 * random.random() * height)
        new_height = height 
        new_width = width
        # if x_min != 0 or x_max != 0:
        #     new_width += 1
        # if y_min != 0 or y_max != 0:
        #     new_height += 1
        # R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s * max(new_width / width, new_height / height))
        # C[0, 2] = -img.shape[1] / 2  + int(0.5 * (x_max - x_min))# x translation (pixels)
        # C[1, 2] = -img.shape[0] / 2  + int(0.5 * (y_max - y_min))# y translation (pixels)
        # C[0, 0] = width / int(width + (x_max + x_min))
        # C[1, 1] = height / int(height + (y_max + y_min))
    else:
        new_height = height 
        new_width = width


    # Translation
    # T = np.eye(3)
    # T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    # T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    T = np.eye(3)
    T[0, 2] = 0.5 * width  # x translation (pixels)
    T[1, 2] = 0.5 * height 

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT

    C = np.eye(3)
    C[0, 2] += x_min # x translation (pixels)
    C[1, 2] += y_min # y translation (pixels)
    C[0, 0] = width / (width + (x_max + x_min*2))
    C[1, 1] = height / (height + (y_max + y_min*2))
    M = C @ M
    # new_height = int(width + (x_max + x_min)) 
    # new_width = int(height + (y_max + y_min))

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(new_width, new_height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(new_width, new_height), borderValue=(114, 114, 114))
        img = cv2.resize(img, (width, height))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base

    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    img_four_points = np.asarray([[0, 0], [width, 0], [width, height], [0, height]])
    img_four_points = np.hstack((img_four_points, np.ones((4, 1))))
    img_four_points = img_four_points @ M.T
    if perspective:
        img_four_points = (img_four_points[:, :2] / img_four_points[:, 2:3])  # rescale
    else:  # affine
        img_four_points = img_four_points[:, :2]
    
    four_edges_new = np.hstack((img_four_points, np.vstack((img_four_points[1:], img_four_points[0])))).reshape(4, 2, 2).tolist()

    # save_poly(img, img_four_points) # for test

    if n:
        # warp points
        #xy = np.ones((n * 4, 3))
        xy = np.ones((n * 8, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2, 5, 6, 7, 8, 9, 10, 11, 12]].reshape(n * 8, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 16)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 16)

        xy = (xy.reshape(n, 8, 2) * np.asarray([width / new_width, height / new_height])).reshape(n, 16)
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]

        landmarks = xy[:, [8, 9, 10, 11, 12, 13, 14, 15]]
        # mask = np.array(targets[:, 5:] > 0, dtype=np.int32)
        # landmarks = landmarks * mask
        # landmarks = landmarks + mask * (mask - 1)
        # print(width, height)
        # print(new_width, new_height)
        # print(landmarks)

        # landmarks = np.asarray(correct_landmark(landmarks, width, height, four_edges_new))

        # landmarks = np.where(landmarks < 0, 0, landmarks)
        # landmarks[:, [0, 2, 4, 6]] = np.where(landmarks[:, [0, 2, 4, 6]] >= width, width - 1, landmarks[:, [0, 2, 4, 6]])
        # landmarks[:, [1, 3, 5, 7]] = np.where(landmarks[:, [1, 3, 5, 7]] >= height, height - 1,landmarks[:, [1, 3, 5, 7]])        

        # old 
        '''
        landmarks = np.where(landmarks < 0, -1, landmarks)
        landmarks[:, [0, 2, 4, 6]] = np.where(landmarks[:, [0, 2, 4, 6]] > width, -1, landmarks[:, [0, 2, 4, 6]])
        landmarks[:, [1, 3, 5, 7]] = np.where(landmarks[:, [1, 3, 5, 7]] > height, -1,landmarks[:, [1, 3, 5, 7]])

        landmarks[:, 0] = np.where(landmarks[:, 1] == -1, -1, landmarks[:, 0])
        landmarks[:, 1] = np.where(landmarks[:, 0] == -1, -1, landmarks[:, 1])

        landmarks[:, 2] = np.where(landmarks[:, 3] == -1, -1, landmarks[:, 2])
        landmarks[:, 3] = np.where(landmarks[:, 2] == -1, -1, landmarks[:, 3])

        landmarks[:, 4] = np.where(landmarks[:, 5] == -1, -1, landmarks[:, 4])
        landmarks[:, 5] = np.where(landmarks[:, 4] == -1, -1, landmarks[:, 5])

        landmarks[:, 6] = np.where(landmarks[:, 7] == -1, -1, landmarks[:, 6])
        landmarks[:, 7] = np.where(landmarks[:, 6] == -1, -1, landmarks[:, 7])

        landmarks[:, 8] = np.where(landmarks[:, 9] == -1, -1, landmarks[:, 8])
        landmarks[:, 9] = np.where(landmarks[:, 8] == -1, -1, landmarks[:, 9])
        '''

        targets[:,5:] = landmarks

        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        box_from_lm = get_box_from_landmark(landmarks, n, height, width)

        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T, box_from_lm=box_from_lm.T)
        
        targets = targets[i]
        targets[:, 1:5] = box_from_lm[i]

    return img, targets


def box_candidates(box1, box2, box_from_lm, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    # w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    w2, h2 = box_from_lm[2] - box_from_lm[0], box_from_lm[3] - box_from_lm[1]
    
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (ar < ar_thr)  # candidates


def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../coco128/'):  # from utils.datasets import *; extract_boxes('../coco128')
    # Convert detection dataset into classification dataset, with one directory per class

    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in img_formats:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../coco128', weights=(0.9, 0.1, 0.0)):  # from utils.datasets import *; autosplit('../coco128')
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    # Arguments
        path:       Path to images directory
        weights:    Train, val, test weights (list)
    """
    path = Path(path)  # images dir
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split
    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path / x).unlink() for x in txt if (path / x).exists()]  # remove existing
    for i, img in tqdm(zip(indices, files), total=n):
        if img.suffix[1:] in img_formats:
            with open(path / txt[i], 'a') as f:
                f.write(str(img) + '\n')  # add image to txt file

# CUDA_VISIBLE_DEVICES="" python train.py --data data/card.yaml --cfg models/yolov5s.yaml --img-size 128 128 --batch-size 64 --epochs 80 --weights yolov5s-face.pt