# 测试检测指针度数
import argparse
import os
import numpy as np
import time
import glob
import random
import math
import cv2

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.utils import make_grid, save_image

# magic number
kernelsize = 3
METER_SHAPE = 512
CIRCLE_CENTER = [256, 256]
CIRCLE_RADIUS = 250
PI = 3.1415926536
LINE_HEIGHT = 120
LINE_WIDTH = 1570  # 任意长度，之后可改成周长
TYPE_THRESHOLD = 30
METER_CONFIG = [{
    'scale_value': 0.4 / 40.0,
    'range': 25.0,
    'unit': "(MPa)"
}, {
    'scale_value': 1.6 / 32.0,
    'range': 1.6,
    'unit': "(MPa)"
}]


def read_process(label_maps):
    label_maps = use_erode_image(label_maps)
    # Convert the circular meter into rectangular meter
    line_images = creat_line_image(label_maps)
    # Convert the 2d meter into 1d meter
    scale_data, pointer_data = convert_1d_data(line_images)
    # Fliter scale data whose value is lower than the mean value
    scale_mean_filtration(scale_data)
    # Get scale_num, scales and ratio of meters
    result = get_meter_reader(scale_data, pointer_data)
    return result


# 腐蚀
def use_erode_image(meter_image, erode_kernel=3):
    kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
    erode_image = cv2.erode(meter_image, kernel)
    return erode_image


# 转矩形
def creat_line_image( meter_image):
    line_image = np.zeros((LINE_HEIGHT, LINE_WIDTH), dtype=np.uint8)
    for row in range(LINE_HEIGHT):
        for col in range(LINE_WIDTH):
            theta = PI * 2 / LINE_WIDTH * (col + 1)
            rho = CIRCLE_RADIUS - row - 1
            x = int(CIRCLE_CENTER[0] + rho * math.cos(theta) + 0.5)
            y = int(CIRCLE_CENTER[1] - rho * math.sin(theta) + 0.5)
            line_image[row, col] = meter_image[x, y]
            # if col <40:  # TODO :为什么line_image不是从theta = 0开始
            #     cv2.circle(meter_image, (x,y), 1, 255, 4)
            #     cv2.circle(line_image, (row, col), 1, 155, 4)
            #     print(x, y, row, col, theta, meter_image[x, y])
    return line_image


# 转1D
def convert_1d_data( meter_image):
    scale_data = np.zeros((LINE_WIDTH), dtype=np.uint8)
    pointer_data = np.zeros((LINE_WIDTH), dtype=np.uint8)
    for col in range(LINE_WIDTH):
        for row in range(LINE_HEIGHT):
            if meter_image[row, col] == 38:
                pointer_data[col] += 1
            elif meter_image[row, col] == 75:
                scale_data[col] += 1
    return scale_data, pointer_data


def scale_mean_filtration( scale_data):
    mean_data = np.mean(scale_data)
    for col in range(LINE_WIDTH):
        if scale_data[col] < mean_data:
            scale_data[col] = 0


def get_meter_reader(scale_data, pointer_data):
    scale_flag = False
    pointer_flag = False
    one_scale_start = 0
    one_scale_end = 0
    one_pointer_start = 0
    one_pointer_end = 0
    scale_location = list()
    pointer_location = 0
    max_pointer_location = 0
    for i in range(LINE_WIDTH - 1):
        if scale_data[i] > 0 and scale_data[i + 1] > 0:
            if scale_flag == False:
                one_scale_start = i
                scale_flag = True
        if scale_flag:
            if scale_data[i] == 0 and scale_data[i + 1] == 0:
                one_scale_end = i - 1
                one_scale_location = (one_scale_start + one_scale_end) / 2
                scale_location.append(one_scale_location)
                one_scale_start = 0
                one_scale_end = 0
                scale_flag = False
        if pointer_data[i] > 0 and pointer_data[i + 1] > 0:
            if pointer_flag == False:
                one_pointer_start = i
                pointer_flag = True
        if pointer_flag:
            if pointer_data[i] == 0 and pointer_data[i + 1] == 0:
                one_pointer_end = i - 1
                # 去除杂点，选取最大指针距离的
                if (one_pointer_end - one_pointer_start) > max_pointer_location:
                    pointer_location = (one_pointer_start + one_pointer_end) / 2
                    max_pointer_location = one_pointer_end - one_pointer_start
                one_pointer_start = 0
                one_pointer_end = 0
                pointer_flag = False

    scale_num = len(scale_location)
    scales = -1
    ratio = -1
    if scale_num > 0:
        for i in range(scale_num - 1):
            if scale_location[i] <= pointer_location and pointer_location < scale_location[i + 1]:
                scales = i + 1 + (pointer_location - scale_location[i]) / (scale_location[i + 1] - scale_location[i] + 1e-05) 
                break
        ratio = (pointer_location - scale_location[0]) / (scale_location[scale_num - 1] - scale_location[0] + 1e-05)
    result = {'scale_num': scale_num, 'scales': scales, 'ratio': ratio}
    return result


def main():

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--in-path', type=str, required=True, help='image to test')
    # parser.add_argument('--out-path', type=str, required=True, help='mask image to save')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--ckpt', type=str, default='deeplab-resnet.pth',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes','invoice'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    model_s_time = time.time()
    model = DeepLab(num_classes=args.num_classes,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model = model.cuda()
    model_u_time = time.time()
    model_load_time = model_u_time-model_s_time
    print("model load time is {}".format(model_load_time))

    composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])
    for name in os.listdir(args.in_path):
        s_time = time.time()
        image = Image.open(args.in_path+"/"+name).convert('RGB')
        # image = Image.open(args.in_path).convert('RGB')
        target = Image.open(args.in_path+"/"+name).convert('L')
        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        model.eval()
        if args.cuda:
            tensor_in = tensor_in.cuda()
        with torch.no_grad():
            output = model(tensor_in)

        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy()),
                                3, normalize=False, range=(0, 255))
        save_image(grid_image, args.in_path+"/"+"{}_mask.png".format(name[0:-4]))
        u_time = time.time()
        img_time = u_time-s_time
        print("image:{} time: {} ".format(name,img_time))
        # save_image(grid_image, args.out_path)
        # print("type(grid) is: ", type(grid_image))
        # print("grid_image.shape is: ", grid_image.shape)
    print("image save in in_path.")

    imgs_paths = glob.glob(args.in_path + '\\*.png')
    for img_path in imgs_paths:
        src_img = cv2.imread(img_path, 0)
        src_img = cv2.resize(src_img, (METER_SHAPE, METER_SHAPE))
        erosion_img = use_erode_image(src_img, 3)
        result = read_process(erosion_img)
        print(result)

        if result['scale_num'] > TYPE_THRESHOLD:
            value = result['scales'] * METER_CONFIG[0]['scale_value']
        else:
            value = result['scales'] * METER_CONFIG[1]['scale_value']
        print("-- Meter result: {} --\n".format(value))



if __name__ == '__main__':
    main()
    
    cv2.waitKey(0)

# python reader_infer.py --in-path E:\sc\image_data\meter\meter_seg\images\test_true 
#                   --ckpt run\meter_seg_voc\deeplab-resnet\model_best.pth.tar --backbone resnet
