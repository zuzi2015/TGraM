from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import random
import math
import time
import copy

import _init_paths
from opts import opts
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            #np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            #np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            #np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            #np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]

        return imw, targets, M
    else:
        return imw


def get_data(img_path, label_path):
    height = 608
    width = 1088
    img = cv2.imread(img_path)  # BGR
    if img is None:
        raise ValueError('File corrupt {}'.format(img_path))

    h, w, _ = img.shape
    img, ratio, padw, padh = letterbox(img, height=height, width=width)

    # Load labels
    if os.path.isfile(label_path):
        labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

        # Normalized xywh to pixel xyxy format
        labels = labels0.copy()
        labels[:, 2] = ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
        labels[:, 3] = ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
        labels[:, 4] = ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
        labels[:, 5] = ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh
    else:
        labels = np.array([])

    plotFlag = False
    if plotFlag:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(50, 50))
        plt.imshow(img[:, :, ::-1])
        plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
        plt.axis('off')
        plt.savefig('test.jpg')
        time.sleep(10)

    nL = len(labels)
    if nL > 0:
        # convert xyxy to xywh
        labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
        labels[:, 2] /= width
        labels[:, 3] /= height
        labels[:, 4] /= width
        labels[:, 5] /= height

    # img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

    return img, labels, img_path, (h, w)


def gaussian2D_bbox(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    y *= 0.1
    x *= 0.1
    # print(y)
    # print(np.shape(y), np.shape(x))
    # print('x * x', np.shape(x * x))
    # print('y * y', np.shape(y * y))
    # print('x*x+y*y', np.shape(x*x+y*y))

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    # print('h shape:', np.shape(h))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian_bbox(heatmap, center, shape, radius, k=1):
    diameter = 2 * radius + 1
    h, w = shape
    h = int(h / 2)
    w = int(w / 2)
    gaussian = gaussian2D_bbox((2 * h + 1, 2 * w + 1), sigma=diameter / 6)
    # gaussian = gaussian2D((6 * h + 1, 6 * w + 1), sigma=diameter / 6)

    # gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    # print(type(gaussian))
    # print(np.shape(gaussian))

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, w), min(width - x, w + 1)
    top, bottom = min(y, h), min(height - y, h + 1)

    # print(width, height)
    # print(left, right, top, bottom)
    # print(w, h)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[h - top:h + bottom, w - left:w + right]
    # masked_gaussian = gaussian[3 * h - top:3 * h + bottom, 3 * w - left:3 * w + right]

    # print(np.shape(masked_heatmap))
    # print(h - top, h + bottom, w - left, w + right, np.shape(masked_gaussian))
    # print(masked_gaussian)
    # import sys
    # sys.exit(0)
    # masked_gaussian = gaussian

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_msra_gaussian_bbox(heatmap, center, shape):
    # tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]

    # ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    # br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    ul = [int(mu_x - shape[0]), int(mu_y - shape[1])]
    br = [int(mu_x + shape[0] + 1), int(mu_y + shape[1] + 1)]

    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap

    # size = 2 * tmp_size + 1
    size = 2 * 5 + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    print(x-x0)

    import sys
    sys.exit(0)


    x = np.arange(0, 2 * shape[0] + 1, 1, np.float32)
    y = np.arange(0, 2 * shape[1] + 1, 1, np.float32)
    x0 = shape[0]
    y0 = shape[1]

    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * shape[0] ** 2))

    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


# def draw_msra_gaussian(heatmap, center, sigma):
#   tmp_size = sigma * 3
#   mu_x = int(center[0] + 0.5)
#   mu_y = int(center[1] + 0.5)
#   w, h = heatmap.shape[0], heatmap.shape[1]
#   ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
#   br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
#   if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
#     return heatmap
#   size = 2 * tmp_size + 1
#   x = np.arange(0, size, 1, np.float32)
#   y = x[:, np.newaxis]
#   x0 = y0 = size // 2
#   g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
#   g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
#   g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
#   img_x = max(0, ul[0]), min(br[0], h)
#   img_y = max(0, ul[1]), min(br[1], w)
#   heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
#     heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
#     g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
#   return heatmap
#
# def draw_umich_gaussian(heatmap, center, radius, k=1):
#     diameter = 2 * radius + 1
#     gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
#
#     x, y = int(center[0]), int(center[1])
#
#     height, width = heatmap.shape[0:2]
#
#     left, right = min(x, radius), min(width - x, radius + 1)
#     top, bottom = min(y, radius), min(height - y, radius + 1)
#
#     masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
#     masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
#     if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
#         np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
#     return heatmap


def hm_gen(img_path, label_path, num_classes, down_ratio, K):
    imgs, labels, img_path, (input_h, input_w) = get_data(img_path, label_path)

    output_h = imgs.shape[0] // down_ratio
    output_w = imgs.shape[1] // down_ratio
    num_objs = labels.shape[0]

    # print('\noutput_h', output_h)
    # print('\noutput_w', output_w)
    # print('\nnum_objs', num_objs)
    # import sys
    # sys.exit(0)

    hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
    # if opt.ltrb:
    #     wh = np.zeros((opt.K, 4), dtype=np.float32)
    # else:
    #     wh = np.zeros((opt.K, 2), dtype=np.float32)
    wh = np.zeros((K, 2), dtype=np.float32)
    reg = np.zeros((K, 2), dtype=np.float32)
    ind = np.zeros((K,), dtype=np.int64)
    reg_mask = np.zeros((K,), dtype=np.uint8)
    ids = np.zeros((K,), dtype=np.int64)
    bbox_xys = np.zeros((K, 4), dtype=np.float32)

    # draw_gaussian = draw_msra_gaussian if opt.mse_loss else draw_umich_gaussian
    draw_gaussian = draw_msra_gaussian
    # draw_gaussian = draw_umich_gaussian
    for k in range(num_objs):
        label = labels[k]
        bbox = label[2:]

        # cls_id = int(label[0])
        cls_id = 0

        bbox[[0, 2]] = bbox[[0, 2]] * output_w
        bbox[[1, 3]] = bbox[[1, 3]] * output_h
        bbox_amodal = copy.deepcopy(bbox)
        bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
        bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
        bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
        bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
        bbox[0] = np.clip(bbox[0], 0, output_w - 1)
        bbox[1] = np.clip(bbox[1], 0, output_h - 1)
        h = bbox[3]
        w = bbox[2]

        bbox_xy = copy.deepcopy(bbox)
        bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
        bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
        bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
        bbox_xy[3] = bbox_xy[1] + bbox_xy[3]

        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            # radius = 6 if opt.mse_loss else radius
            # radius = max(1, int(radius)) if self.opt.mse_loss else radius
            ct = np.array(
                [bbox[0], bbox[1]], dtype=np.float32)
            ct_int = ct.astype(np.int32)

            radius *= 3

            draw_umich_gaussian_bbox(hm[cls_id], ct_int, (math.ceil(h), math.ceil(w)), radius)
            # draw_msra_gaussian_bbox(hm[cls_id], ct_int, (math.ceil(h), math.ceil(w)))
            # draw_gaussian(hm[cls_id], ct_int, radius)

            # if opt.ltrb:
            #     wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
            #             bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
            # else:
            #     wh[k] = 1. * w, 1. * h
            wh[k] = 1. * w, 1. * h
            ind[k] = ct_int[1] * output_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1
            ids[k] = label[1]
            bbox_xys[k] = bbox_xy

    ret = {'input': imgs, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids,
           'bbox': bbox_xys}
    return ret


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    # mask[np.where(mask == 0)] = 0.25
    # print(mask)
    mask *= 0.9
    print(mask)
    print(type(mask), np.shape(mask))
    print('min:', np.min(mask), ', max:', np.max(mask))
    mask = 1-mask

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # print('\nheatmap', np.shape(heatmap))
    # print(heatmap)
    # print(image)
    # cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('hm_2')), heatmap*255)#ret['hm'][0] * 255)
    # import sys
    # sys.exit(0)

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image) / 255
    return norm_image(cam), norm_image(heatmap)


def main(img_path, label_path, save_dir, down_ratio, K):
    ret = hm_gen(img_path=img_path, label_path=label_path, num_classes=1, down_ratio=down_ratio, K=K)

    h, w = ret['input'].shape[1], ret['input'].shape[2]
    with open("/workspace/fairmot/result/cam/a.txt", "w") as f:
        f.write(str(ret['hm'][0]))
    print('\n', np.sort(ret['hm'][0]))
    print('\nret[hm]:', np.shape(ret['hm']))
    print('\nret[input]:', np.shape(ret['input']))

    cam, heatmap = gen_cam(ret['input'], ret['hm'][0])
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('0_cam')), cam)
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('0_heatmap')), heatmap)


if __name__ == '__main__':
    # labels0 = np.loadtxt("/workspace/fairmot/src/data/MOT17/labels_with_ids/train/MOT17-02-FRCNN/img1/000006.txt",
    #                       dtype=np.float32).reshape(-1, 6)
    # print(labels0)
    # encoding = 'gbk',


    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    # opt = opts().init()

    img_path = "/workspace/fairmot/src/data/AIR_MOT/images/train/Sydney_11/img/000001.jpg"
    label_path = \
        "/workspace/fairmot/src/data/AIR_MOT/labels_with_ids/train/Sydney_11/img/000001.txt"
    save_dir = '/workspace/fairmot/result/cam/'

    main(img_path=img_path, label_path=label_path, save_dir=save_dir, down_ratio=1, K=256)

