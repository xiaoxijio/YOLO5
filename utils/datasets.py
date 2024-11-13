import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import xyxy2xywh

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


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
                      rank=-1, world_size=1, workers=8):
    dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                  augment=augment,  # 增强图像
                                  hyp=hyp,  # 超参数
                                  rect=rect,  # 矩形训练
                                  cache_images=cache,  # 缓存
                                  single_cls=opt.single_cls,  # 单类别用到
                                  stride=int(stride),  # 从输入到输出 下采样的比例
                                  pad=pad,
                                  rank=rank)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])

    dataloader = InfiniteDataLoader(dataset,
                                    batch_size=batch_size,
                                    num_workers=nw,
                                    sampler=None,
                                    pin_memory=True,
                                    collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
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
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, rank=-1):
        try:
            f = []  # 存储图像文件路径
            for p in path if isinstance(path, list) else [path]:
                p = str(Path(p))  # 将路径转化为适合当前操作系统的格式
                parent = str(Path(p).parent) + os.sep  # 获取路径的父目录，并添加路径分隔符 os.sep
                if os.path.isfile(p):  # 如果 p 是文件路径
                    with open(p, 'r') as t:  # 将每行内容转成绝对路径
                        t = t.read().splitlines()  # 把以 './' 开头的路径替换为完整的父目录路径，确保路径完整
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
                elif os.path.isdir(p):  # 如果 p 是文件夹路径
                    f += glob.iglob(p + os.sep + '*.*')  # 遍历文件夹中的所有文件
                else:
                    raise Exception('%s 不存在' % p)
            self.img_files = sorted(  # 过滤并排序图像文件
                [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats])
        except Exception as e:
            raise Exception('加载数据错误 %s: %s\n' % (path, e))

        n = len(self.img_files)
        assert n > 0, '没有找到图像 %s.' % path
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch索引
        nb = bi[-1] + 1  # 一个 epoch有多少个 batch
        # 比如我有105个数据, batch_size为16, 那么我一共有 7个batch 32-47图片的batch索引为2

        self.n = n  # 图片数量
        self.batch = bi  # batch索引
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # 一次加载4张图像到马赛克中（仅在训练期间）
        self.mosaic_border = [-img_size // 2, -img_size // 2]  # 限定范围
        self.stride = stride  # 下采样总值

        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/
        self.label_files = [x.replace(sa, sb, 1).replace(os.path.splitext(x)[-1], '.txt') for x in self.img_files]

        cache_path = str(Path(self.label_files[0]).parent) + '.cache'  # 缓存路径
        if os.path.isfile(cache_path):  # 之前有缓存
            cache = torch.load(cache_path)  # 加载
            if cache['hash'] != get_hash(self.label_files + self.img_files):  # 数据发生了变化
                cache = self.cache_labels(cache_path)  # 重新缓存
        else:
            cache = self.cache_labels(cache_path)  # 缓存

        labels, shapes = zip(*[cache[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)

        if self.rect:  # 矩形训练  https://github.com/ultralytics/yolov3/issues/232
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

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # 数字缺失、发现、空、数据集、重复 (记录错误数据量)
        pbar = enumerate(self.label_files)
        pbar = tqdm(pbar)

        for i, file in pbar:
            l = self.labels[i]  # 标签
            if l is not None and l.shape[0]:
                assert l.shape[1] == 5, '> 5 label columns: %s' % file  # 五个特征数据是否完整
                assert (l >= 0).all(), 'negative labels: %s' % file  # 标签值是否大于0
                assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file  # 归一化
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # 计算重复的框
                    nd += 1  # 去重
                if single_cls:
                    l[:, 0] = 0  # 单个类别，标签值全置为 0
                self.labels[i] = l
                nf += 1
            else:
                ne += 1

            pbar.desc = '标签扫描结果 %s (%g found, %g 缺失, %g 空, %g 重复, for %g images)' % (
                cache_path, nf, nm, ne, nd, n)

        if nf == 0:
            s = '未发现标签数据 %s.' % (os.path.dirname(file) + os.sep)
            print(s)
            assert not augment, '%s. 没有标签没法训练.' % s

        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = '缓存图片 (%.1fGB)' % (gb / 1E9)

    def cache_labels(self, path='labels.cache'):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                image = Image.open(img)
                image.verify()  # PIL verify
                shape = exif_size(image)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
                if len(l) == 0:
                    l = np.zeros((0, 5), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                x[img] = [None, None]
                print('WARNING: %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:  # 马赛克数据增强
            img, labels = load_mosaic(self, index)
            shapes = None
        else:
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

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

        if self.augment:
            if not mosaic:  # 这个之前在 mosaic方法最后做过了
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])
            # h:色调 s:饱和度 V:亮度
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

        nL = len(labels)  # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416 要满足pytorch的格式
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640):
        self.mode = 'images'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


class LoadImages:  # for inference
    def __init__(self, path, img_size=640):
        p = str(Path(path))  # os-agnostic
        # p = os.path.abspath(p)  # absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist' % p)

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (p, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nf, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nf, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


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


def get_hash(files):
    # 返回文件列表的单个哈希值
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def load_mosaic(self, index):
    """将四张图片拼在一起"""
    labels4 = []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # 随机中心点位置
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 随机找 3个图片index

    for i, index in enumerate(indices):
        img, _, (h, w) = load_image(self, index)  # 加载随机找的图片

        if i == 0:  # 左上
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # 初始化大图 (拼接四张图的大图)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # 小图片越界选大图为顶点
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # 小图片越界就剪掉
        elif i == 1:  # 右上
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # 左下
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # 右下
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # 小图放入大图
        padw = x1a - x1b  # 由于小图可能填充不满，所以还需要计算差异值，因为一会要更新坐标框标签
        padh = y1a - y1b

        # Labels 标签值要重新计算，因为现在都放到大图中了
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # 小图 xywh 相对到大图 x1y1x2y2
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    if len(labels4):  # labels 坐标计算完之后可能越界，调整坐标值，让他们都在大图中
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])

    # 对整合的大图再进行随机旋转、平移、缩放、裁剪
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)

    return img4, labels4


def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2  # 为了支持图像的平移，设置了一个边框尺寸，以防止平移后目标框超出边界

    # 旋转 平移 缩放等操作 都需要系数矩阵
    C = np.eye(3)  # 中心化矩阵 C：将图像中心移到 (0,0)，方便后续变换操作
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    R = np.eye(3)  # 旋转和缩放矩阵 R：通过随机角度和缩放比例生成一个旋转缩放矩阵
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    S = np.eye(3)  # 剪切矩阵 S：随机剪切角度，生成剪切变换矩阵
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    T = np.eye(3)  # 平移矩阵 T：生成一个平移矩阵，使图像随机向上下左右平移
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # 组合旋转矩阵
    M = T @ S @ R @ C  # 操作顺序（从右到左）很重要
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # 变换标签坐标
    n = len(targets)
    if n:
        # 变换目标框的四个角点：将目标框的四个角点按照综合变换矩阵 M 进行坐标变换
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = xy[:, :2].reshape(n, 8)

        # 生成新的边界框：根据变换后的四个角点计算新的最小边界框
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # 剪裁边界框：确保新的边界框在图像边界内
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # 过滤候选框：过滤无效的候选框，确保只有合理的目标框被保留。
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets  # 返回处理后的图像和新的目标标签


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    """
    筛选增强后的目标框 box2 是否满足一定的条件，从而决定该目标框是否为有效候选框
    :param box1: 原始框的坐标（未增强），维度是 (4, n)，即 [x1, y1, x2, y2]
    :param box2: 增强后的目标框坐标，格式和 box1 一致
    :param wh_thr: 宽高的最小阈值。此参数确保候选框的宽高不会过小
    :param ar_thr: 宽高比的最大阈值。此参数限制候选框的长宽比例，避免极度细长的框
    :param area_thr: 面积比例的最小阈值。此参数保证增强后的框相对于原始框的面积变化不过大
    :return:
    """
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]  # 原始框的宽和高
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]  # 增强后框的宽和高
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # 纵横比
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # 过滤


def load_image(self, index):
    """ 从数据集中加载 1张图像，返回图像，原始图像，调整大小的图像 """
    img = self.imgs[index]
    if img is None:  # 没有缓存
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # 原始 hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # 只有在训练时才调整大小
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized
