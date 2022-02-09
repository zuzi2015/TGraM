from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch

import _init_paths
from PIL import Image
from opts import opts
from lib.models.model import create_model, load_model
from pytorch_grad_cam import GradCAM, \
                              ScoreCAM, \
                              GradCAMPlusPlus, \
                              AblationCAM, \
                              XGradCAM, \
                              EigenCAM, \
                              EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                          deprocess_image, \
                                          preprocess_image


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


def read_img(paths, opt):
    imgs = []
    for i, img_path in enumerate(paths):
        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

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


def main(opt, method, exp, paths, aug_smooth=False, eigen_smooth=False):
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

    # print(model)
    # import sys
    # sys.exit(0)
    # target_layer = model.p_fcn[-1]
    target_layer = model.ida_up.node_2

    methods = \
        {"gradcam": GradCAM,
         # "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         # "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}
    cam = methods[method](model=model,
                          target_layer=target_layer,
                          use_cuda=use_cuda)

    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    # rgb_img = np.float32(rgb_img) / 255
    # input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    input_tensor, rgb_img = read_img(paths, opt)
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = input_tensor.cuda().unsqueeze(0)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 1

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        aug_smooth=aug_smooth,
                        eigen_smooth=eigen_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    #===============================================================================
    grayscale_cam = 1 - grayscale_cam

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
    gb = gb_model(input_tensor, target_category=target_category)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cam_image = Image.fromarray(cam_image.astype(np.uint8))
    cam_image.save(os.path.join(opt.output_dir, 'cam', f'{exp}_{method}_2_cam.jpg'))
    gb = Image.fromarray(gb.astype(np.uint8))
    gb.save(os.path.join(opt.output_dir, 'cam', f'{exp}_{method}_gb.jpg'))
    cam_gb = Image.fromarray(cam_gb.astype(np.uint8))
    cam_gb.save(os.path.join(opt.output_dir, 'cam', f'{exp}_{method}_cam_gb.jpg'))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    # cam(opt)
    main(opt=opt,
         method='gradcam++',
         exp='20210530_mot17',
         paths=["/workspace/fairmot/src/data/MOT17/images/train/MOT17-02-DPM/img1/000018.jpg",
                "/workspace/fairmot/src/data/MOT17/images/train/MOT17-02-DPM/img1/000017.jpg",
                "/workspace/fairmot/src/data/MOT17/images/train/MOT17-02-DPM/img1/000016.jpg"],
         aug_smooth=False,
         eigen_smooth=False)
