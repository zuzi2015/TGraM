from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import _init_paths
from opts import opts
from lib.models.model import create_model, load_model

from cam_torchcam import hm_gen


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


def read_img(paths, opt, save_dir):
    imgs = []
    for i, img_path in enumerate(paths):
        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('0_origin')), img0)

        # Padded resize
        img, _, _, _ = letterbox(img0, height=opt.img_size[1], width=opt.img_size[0])

        if i == 0:
            img_cur = img

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        imgs.append(img)

    return torch.Tensor(imgs), img_cur


def norm_image(tensor):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = np.sum(abs(tensor[0].cpu().detach().numpy()), axis=0)
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)

    image = cv2.applyColorMap(np.uint8(255 * image), cv2.COLORMAP_JET)
    image = np.float32(image) / 255
    image = image[..., ::-1]  # gbr to rgb

    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def norm_fuse_image(tensors):
    image = np.sum(abs(tensors[0][0].cpu().detach().numpy()), axis=0)
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    for tensor in tensors[1:]:
        img = np.sum(abs(tensor[0].cpu().detach().numpy()), axis=0)
        img = img.copy()
        img -= np.max(np.min(img), 0)
        img /= np.max(img)
        image += img

    image = cv2.applyColorMap(np.uint8(255 * image), cv2.COLORMAP_JET)
    image = np.float32(image) / 255
    image = image[..., ::-1]  # gbr to rgb

    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def norm_mul_image(tensors):
    image = np.sum(abs(tensors[0][0].cpu().detach().numpy()), axis=0)
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image += 1e-3
    for tensor in tensors[1:]:
        img = np.sum(abs(tensor[0].cpu().detach().numpy()), axis=0)
        img = img.copy()
        img -= np.max(np.min(img), 0)
        img /= np.max(img)
        img += 1e-3
        image *= img

    image = cv2.applyColorMap(np.uint8(255 * image), cv2.COLORMAP_JET)
    image = np.float32(image) / 255
    image = image[..., ::-1]  # gbr to rgb

    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def norm_mul_image_mask(tensors, mask):
    image = np.sum(abs(tensors[0][0].cpu().detach().numpy()), axis=0)
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)

    # image *= mask

    image += 1e-4 #1e-3
    for tensor in tensors[1:]:
        img = np.sum(abs(tensor[0].cpu().detach().numpy()), axis=0)
        img = img.copy()
        img -= np.max(np.min(img), 0)
        img /= np.max(img)
        img += 1e-3
        image *= img

    # image += (mask * 0.01)

    mask[mask > 0.9] *= 2
    mask[mask < 0.01] *= 0

    image = image * mask #+ image * 0.05

    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image += 1e-3

    image[image < 0.00002] *= 0.1
    image[(image < 0.1) * (image > 0.02)] *= 2
    image[image > 0.7] *= 0.2


    image = cv2.applyColorMap(np.uint8(255 * image), cv2.COLORMAP_JET)
    image = np.float32(image) / 255
    image = image[..., ::-1]  # gbr to rgb

    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def main(opt, paths, save_dir, label_path):
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
        use_cuda = True
    else:
        opt.device = torch.device('cpu')
        use_cuda = False
    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()

    input_tensor, rgb_img = read_img(paths, opt, save_dir)
    # rgb_img = np.float32(rgb_img) / 255
    if use_cuda:
        input_tensor = input_tensor.cuda().unsqueeze(0)

    # x[0], x[1], y[-2], y[0]
    _, feat_ida, feat_stere, x0, x1, y_2, y0, bb0, bb1, bb2, bb3, bb4, bb5 = model(input_tensor[0])
    # print('\ninput', np.shape(input_tensor))
    img_size = input_tensor.shape[3:]
    # print(img_size)
    # import sys
    # sys.exit(0)

    feat_ida = F.interpolate(feat_ida, size=img_size, mode="bilinear", align_corners=True)
    feat_stere = F.interpolate(feat_stere, size=img_size, mode="bilinear", align_corners=True)

    x0 = F.interpolate(x0, size=img_size, mode="bilinear", align_corners=True)
    x1 = F.interpolate(x1, size=img_size, mode="bilinear", align_corners=True)
    y1 = F.interpolate(y_2, size=img_size, mode="bilinear", align_corners=True)
    y0 = F.interpolate(y0, size=img_size, mode="bilinear", align_corners=True)
    bb0 = F.interpolate(bb0, size=img_size, mode="bilinear", align_corners=True)
    bb1 = F.interpolate(bb1, size=img_size, mode="bilinear", align_corners=True)
    bb2 = F.interpolate(bb2, size=img_size, mode="bilinear", align_corners=True)
    bb3 = F.interpolate(bb3, size=img_size, mode="bilinear", align_corners=True)
    bb4 = F.interpolate(bb4, size=img_size, mode="bilinear", align_corners=True)
    bb5 = F.interpolate(bb5, size=img_size, mode="bilinear", align_corners=True)
    # print('\nfeat_ida', np.shape(np.sum(feat_ida[0].cpu().detach().numpy(), axis=0)))
    # print('\nfeat_stere', np.shape(feat_stere))
    img_ida = norm_image(feat_ida)
    img_stere = norm_image(feat_stere)

    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_ida_fairmot_1')), img_ida)
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_stere_fairmot_1')), img_stere)

    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_y2')), img_ida)
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_x2')), img_stere)
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_x0')), norm_image(x0))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_x1')), norm_image(x1))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_y1')), norm_image(y1))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_y0')), norm_image(y0))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0')), norm_image(bb0))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb1')), norm_image(bb1))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb2')), norm_image(bb2))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb3')), norm_image(bb3))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb4')), norm_image(bb4))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb5')), norm_image(bb5))

    y2 = feat_ida
    x2 = feat_stere
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_y1')), norm_fuse_image([y1, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_y2')), norm_fuse_image([y2, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_y0')), norm_fuse_image([y0, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_y1_y2')), norm_fuse_image([y1, y2, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_x0')), norm_fuse_image([x0, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_x0_y0')), norm_fuse_image([y0, x0, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_y0_y0')), norm_fuse_image([y0, y0, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_y0_y1')), norm_fuse_image([y0, y1, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_y1_y1')), norm_fuse_image([y1, y1, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_x0_x0')), norm_fuse_image([x0, x0, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_x0_y1')), norm_fuse_image([x0, y1, bb0]))

    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_y1_mul')), norm_mul_image([y1, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_y2_mul')), norm_mul_image([y2, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_y0_mul')), norm_mul_image([y0, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_y1_y2_mul')), norm_mul_image([y1, y2, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_x0_mul')), norm_mul_image([x0, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_x0_y0_mul')), norm_mul_image([y0, x0, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_y0_y0_mul')), norm_mul_image([y0, y0, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_y0_y1_mul')), norm_mul_image([y0, y1, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_y1_y1_mul')), norm_mul_image([y1, y1, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_x0_x0_mul')), norm_mul_image([x0, x0, bb0]))
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('feat_fairmot_bb0_x0_y1_mul')), norm_mul_image([x0, y1, bb0]))

    ret = hm_gen(img_path=paths[0], label_path=label_path, num_classes=1, down_ratio=1, K=256)
    mask = ret['hm'][0]
    print(np.min(mask), np.max(mask))
    # norm_mul_image_mask(tensors, mask)
    cv2.imwrite(os.path.join(save_dir, '{}.jpg'.format('0_mask')),
                # norm_mul_image_mask([x0, x0, bb0], mask)) # 备选
                norm_mul_image_mask([bb0, x0, x0], mask))
                # norm_mul_image([x0, x0, x0, bb0]))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # opt = opts().init()
    #
    # # cam(opt)
    # main(opt=opt,
    #      paths=["/workspace/fairmot/src/data/AIR_MOT/images/train/Sydney_11/img/000001.jpg"],
    #             # "/workspace/fairmot/src/data/MOT17/images/train/MOT17-02-DPM/img1/000017.jpg",
    #             # "/workspace/fairmot/src/data/MOT17/images/train/MOT17-02-DPM/img1/000016.jpg"],
    #      save_dir='/workspace/fairmot/result/cam/Sydney_11',
    #      label_path="/workspace/fairmot/src/data/AIR_MOT/labels_with_ids/train/Sydney_11/img/000001.txt")


    from model_resources.flops import compute_gflops
    from model_resources.num_parameters import count_parameters

    arch = 'tgrammbv3' #'tgrammbv3'  'dla_34'  'tgrammbseg'
    heads = {'hm': 2, 'wh': 4, 'id': 128, 'reg': 2}
    head_conv = 256

    model = create_model(arch, heads, head_conv)
    # model = load_model(model, opt.load_model)
    model = model.to('cuda')
    model.eval()

    # flops = compute_gflops(model, in_shape=(1, 3, 1088, 608), tasks=None)
    flops = compute_gflops(model, in_shape=([2, 3, 3, 1088, 608]), tasks=None)
    params = count_parameters(model)

    print('Params:', params / 1e6, 'M FLOPs:', flops / 2, 'G')

