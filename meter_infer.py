# 测试检测指针度数
import numpy as np
import argparse
import os
import os.path as osp
import time
import glob
import math
import cv2
from modeling.deeplab import *
from dataloaders import custom_transforms as tr
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import json
from torchvision import transforms

def rectangle(src, aabb1, aabb2, fill=None, outline=None, width=0):
    if outline is not None:
        outline = tuple(outline)
    if fill is not None:
        fill = tuple(fill)

    dst = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.ImageDraw(dst)

    y1, x1 = aabb1
    y2, x2 = aabb2
    draw.rectangle( xy=(x1, y1, x2, y2), fill=fill, outline=outline, width=width )

    return np.array(dst)

def text_size(text, size, font_path=None):
    font = PIL.ImageFont.truetype(font=font_path, size=size)
    lines = text.splitlines()
    n_lines = len(lines)
    longest_line = max(lines, key=len)
    width, height = font.getsize(longest_line)
    return height * n_lines, width

def draw_text(src, yx, text, size, color=(0, 0, 0), font_path=None):
    dst = PIL.Image.fromarray(src)
    draw = PIL.ImageDraw.ImageDraw(dst)
    y1, x1 = yx
    color = tuple(color)
    font = PIL.ImageFont.truetype(font=font_path, size=size)
    draw.text(xy=(x1, y1), text=text, fill=color, font=font)
    return np.array(dst)

def label_colormap(n_label=256, value=None):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    for i in range(0, n_label):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b

    if value is not None:
        rgb = cmap.reshape(1, -1, 3)
        hsv = PIL.Image.fromarray(rgb, mode="RGB")
        hsv = hsv.convert("HSV")
        hsv = np.array(hsv)
        
        
        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        rgb = PIL.Image.fromarray(hsv, mode="HSV")
        rgb = rgb.convert("RGB")
        cmap = np.array(rgb).reshape(-1, 3)
      
    return cmap

def label2rgb(label,img=None,alpha=0.5,label_names=None,font_size=30,thresh_suppress=0,colormap=None,loc="centroid",font_path=None,):
    if colormap is None:
        colormap = label_colormap()

    res = colormap[label]

    random_state = np.random.RandomState(seed=1234)

    mask_unlabeled = label < 0
    res[mask_unlabeled] = random_state.rand(*(mask_unlabeled.sum(), 3)) * 255

    if img is not None:
        if img.ndim == 2:
            img = img[:, :, None].repeat(3, axis=2)

        res = (1 - alpha) * img.astype(float) + alpha * res.astype(float)
        res = np.clip(res.round(), 0, 255).astype(np.uint8)

    if label_names is None:
        return res

    unique_labels = np.unique(label)
    unique_labels = unique_labels[unique_labels != -1]
    unique_labels = [l for l in unique_labels if label_names[l] is not None]
    if len(unique_labels) == 0:
        return res

    if loc == "centroid":
        for label_i in unique_labels:
            mask = label == label_i
            if 1.0 * mask.sum() / mask.size < thresh_suppress:
                continue
            y, x = np.array(_center_of_mass(mask), dtype=int)

            if label[y, x] != label_i:
                Y, X = np.where(mask)
                point_index = np.random.randint(0, len(Y))
                y, x = Y[point_index], X[point_index]

            text = label_names[label_i]
            height, width = text_size( text, size=font_size, font_path=font_path )
            print(res[y, x])
            gray = PIL.Image.fromarray(np.asarray(res[y, x], dtype=np.uint8).reshape(1, 1, 3)).convert("L")

            if np.array(gray).sum() > 170:
                color = (0, 0, 0)
            else:
                color =(255, 255, 255)
            
            res = draw_text(res, yx=(y-height//2, x-width//2), text=text, color=color, size=font_size, font_path=font_path, )
    elif loc in ["rb", "lt"]:
        text_sizes = np.array([text_size(label_names[l], font_size, font_path=font_path) for l in unique_labels] )
        text_height, text_width = text_sizes.max(axis=0)
        legend_height = text_height * len(unique_labels) + 5
        legend_width = text_width + 20 + (text_height - 10)

        height, width = label.shape[:2]
        legend = np.zeros((height, width, 3), dtype=np.uint8)
        if loc == "rb":
            aabb2 = np.array([height - 5, width - 5], dtype=float)
            aabb1 = aabb2 - (legend_height, legend_width)
        elif loc == "lt":
            aabb1 = np.array([5, 5], dtype=float)
            aabb2 = aabb1 + (legend_height, legend_width)
        else:
            raise ValueError("unexpected loc: {}".format(loc))
        legend = rectangle(legend, aabb1, aabb2, fill=(255, 255, 255))
        alpha = 0.5
        y1, x1 = aabb1.round().astype(int)
        y2, x2 = aabb2.round().astype(int)
        res[y1:y2, x1:x2] = (alpha * res[y1:y2, x1:x2] + alpha * legend[y1:y2, x1:x2])
        for i, l in enumerate(unique_labels):
            box_aabb1 = aabb1 + (i * text_height + 5, 5)
            box_aabb2 = box_aabb1 + (text_height - 10, text_height - 10)
            res = rectangle(res, aabb1=box_aabb1, aabb2=box_aabb2, fill=colormap[l])
            res = draw_text(res, yx=aabb1 + (i * text_height, 10 + (text_height - 10)), text=label_names[l], size=font_size, font_path=font_path, )
    else:
        raise ValueError("unsupported loc: {}".format(loc))

    return res


np.set_printoptions(threshold=np.inf)

import matplotlib
font_path = osp.join(osp.dirname(matplotlib.__file__), "mpl-data/fonts/ttf", "DejaVuSansMono.ttf")

class meter_mask_info():
    def __init__(self, _mask, _dict=None):
        H, W = _mask.shape
        self.circle_center = [H/2, W/2]
        self.circle_radius = min(H, W) / 2
        self.line_height = int(self.circle_radius*0.6)
        self.line_width = int(self.circle_radius*2*math.pi)
        self.kernel_size = 3
        self.label_maps = _mask
        self.infos_dict = _dict
        
    def reading_process(self, ):
        # Normalizing and corrosion semantic map
        norm_images = self.norm_erode_image(self.label_maps)
        # Convert the circular meter into rectangular meter
        line_images = self.creat_line_image(norm_images)
        # Convert the 2d meter into 1d meter
        scale_data, pointer_data = self.convert_1d_data(line_images)
        # Fliter scale data whose value is lower than the mean value
        self.scale_mean_filtration(scale_data)
        # Get scale_num, scales and ratio of meters
        result = self.get_meter_reader(scale_data, pointer_data)
        #PIL.Image.fromarray(label_colormap()[line_images]).save(path)
        # fix scale
        result = self.fix_initial_scale(result)
        value = self.get_value(result)
        return value
        
    def norm_erode_image(self, meter_image):
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        erode_image = cv2.erode(meter_image, kernel)
        return erode_image

    def creat_line_image(self, meter_image):
        line_image = np.zeros((self.line_height, self.line_width), dtype=np.uint8)
        for row in range(self.line_height):
            for col in range(self.line_width):
                theta = math.pi * 2 / self.line_width * (col + 1)
                rho = self.circle_radius - row - 1
                x = int(self.circle_center[0] + rho * math.cos(theta) + 0.5)
                y = int(self.circle_center[1] - rho * math.sin(theta) + 0.5)
                line_image[row, col] = meter_image[x, y]
        return line_image

    def convert_1d_data(self, meter_image):
        scale_data = np.zeros((self.line_width), dtype=np.uint8)
        pointer_data = np.zeros((self.line_width), dtype=np.uint8)
        for col in range(self.line_width):
            for row in range(self.line_height):
                if meter_image[row, col] == 1:
                    pointer_data[col] += 1
                elif meter_image[row, col] == 2:
                    scale_data[col] += 1
        return scale_data, pointer_data

    def scale_mean_filtration(self, scale_data):
        mean_data = np.mean(scale_data)
        for col in range(self.line_width):
            if scale_data[col] < mean_data:
                scale_data[col] = 0
    
    def get_meter_reader(self, scale_data, pointer_data):
        scale_flag = False
        pointer_flag = False
        one_scale_start = 0
        one_scale_end = 0
        one_pointer_start = 0
        one_pointer_end = 0
        scale_location = list()
        scale_width = list()
        scale_width_mean = 0  # 刻度平均宽
        scale_width_first = 0  # 第一个刻度宽
        scale_range = list()  # 刻度之间的距离
        scale_range_median = 0  # 刻度间隔中位数
        pointer_location = 0
        max_pointer_location = 0
        for i in range(self.line_width - 1):
            if scale_data[i] > 0 and scale_data[i + 1] > 0:
                if scale_flag == False:
                    one_scale_start = i
                    scale_flag = True
            if scale_flag:
                if scale_data[i] == 0 and scale_data[i + 1] == 0:
                    one_scale_end = i - 1
                    one_scale_location = (one_scale_start + one_scale_end) / 2
                    scale_location.append(one_scale_location)
                    scale_width.append(one_scale_end - one_scale_start)
                    if len(scale_location) > 1:  # 不计算第一个刻度的长度
                        scale_width_mean = (scale_width_mean + (one_scale_end - one_scale_start))/2
                    else:
                        scale_width_first = one_scale_end - one_scale_start
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

        for i in range(len(scale_location)-1):  # 求刻度间隔
            scale_range.append(scale_location[i+1] - scale_location[i])
        scale_range_median = np.median(scale_range)
        for i in range(len(scale_width)-1):  # 寻找合并的刻度,默认scale_range插入
            if scale_width[i+1] > scale_width_mean*2.8:
                # print(i+1, scale_width[i+1], scale_width_mean)
                merge_num = int(scale_width[i+1]/scale_width_mean)
                if merge_num == 2:  # 最多修复2个刻度合并情况，不然就用角度计算
                    # print('fix merge scale')
                    scale_location.pop(i+1)
                    for j in range(merge_num):
                        scale_location.insert(i+1+j, scale_location[i]+scale_range_median*(j+1))
                    break

        scale_num = len(scale_location)
        scales = -1
        ratio = -1
        ratio_add = -1
        onescalerange_ratio = -1
        if scale_num > 0:
            for i in range(scale_num - 1):
                if scale_location[i] <= pointer_location and pointer_location < scale_location[i + 1]:
                    scales = i + (pointer_location - scale_location[i]) / (scale_location[i + 1] - scale_location[i] + 1e-05) 
                    break
            ratio = (pointer_location - scale_location[0]) / (scale_location[scale_num - 1] - scale_location[0] + 1e-05)
            # 模拟计算首刻度有问题情况下的角度，用不用之后判断
            ratio_add = (pointer_location - scale_location[1] + scale_range_median*2) / (
                         scale_location[scale_num - 1] - scale_location[1] + scale_range_median*2 + 1e-05)
            onescalerange_ratio = (scale_location[scale_num - 1] - scale_location[0] + 1e-05) / self.infos_dict["scale_num"]/(
                                   scale_location[scale_num - 1] - scale_location[0] + 1e-05)
        result = {'scale_num': scale_num, 'scales': scales, 'ratio': ratio,
                  'scale_width_mean': scale_width_mean,
                  'scale_width_first': scale_width_first,
                  'scale_range_median': scale_range_median,
                  'ratio_add': ratio_add,
                  'onescalerange_ratio': onescalerange_ratio}
        # print(result)
        return result

    def fix_initial_scale(self, result):
        if (self.infos_dict["scale_num"] - result['scale_num']) > 2:  # 如果检测刻度出错，直接用ratio
            result['ratio'] = result['ratio'] + result['onescalerange_ratio']
            return result
        else:
            result['ratio'] = result['ratio_add']
        if ((result['scale_width_first'] > result['scale_width_mean']*1.5) and (
            result['scale_num'] < self.infos_dict["scale_num"])):  # 如果第一个刻度过宽，说明第一二个刻度合并
            # print('add initial scale.')
            result['scale_num'] = result['scale_num'] + 1
            result['scales'] = result['scales'] + 1
        else:
            if (self.infos_dict["scale_num"] - result['scale_num']) == 1:  # 没有过宽，但确实合并了
                # print('add 1')
                result['scale_num'] = result['scale_num'] + 1
                result['scales'] = result['scales'] + 1
                return result
        if (self.infos_dict["scale_num"]%2) == 0:  # 如果缺少第一个刻度
            # print('lach 1 scale')
            result['scales'] = result['scales'] + 1
        return result

    def get_value(self, result):
        if result['scale_num'] == self.infos_dict["scale_num"]:  # 如果刻度相等
            print('all scales are checked')
            value = result['scales'] * (self.infos_dict["scale_max"] / (self.infos_dict["scale_num"] - 1))
        else:  # 如果不等，说明缺少。
            print('lack of scale,use ratio')
            value = result['ratio'] * self.infos_dict["scale_max"]
        if value < 0.0001:
            value = 0
        return value

def load_json(path):
    with open(path,"r") as f:
        d=json.load(f)
    return d

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument("--in_path", type=str, required=True, help="image to test")
    parser.add_argument("--out_path", type=str, required=True, help="mask image to save")
    parser.add_argument("--backbone", type=str, default="resnet", choices=["resnet", "xception", "drn", "mobilenet"], help="backbone name (default: resnet)")
    parser.add_argument("--ckpt", type=str, default="deeplab-resnet.pth", help="saved model")
    parser.add_argument("--out_stride", type=int, default=16, help="network output stride (default: 8)")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--gpu_ids", type=str, default="0", help="use which gpu to train, must be a comma-separated list of integers only (default=0)")
    parser.add_argument("--dataset", type=str, default="pascal", choices=["pascal", "coco", "cityscapes","invoice"], help="dataset name (default: pascal)")
    parser.add_argument("--crop_size", type=int, default=513, help="crop image size")
    parser.add_argument("--num_classes", type=int, default=3, help="crop image size")
    parser.add_argument("--sync_bn", type=bool, default=None, help="whether to use sync bn (default: auto)")
    parser.add_argument("--freeze_bn", type=bool, default=False, help="whether to freeze bn parameters (default: False)")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    

    if not osp.exists(args.out_path):
        os.makedirs(args.out_path)
    
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(",")]
        except ValueError:
            raise ValueError("Argument --gpu_ids must be a comma-separated list of integers only")

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
    map_dict = load_json(r"mapping.json")
    model_s_time = time.time()
    model = DeepLab(num_classes=args.num_classes, backbone=args.backbone, output_stride=args.out_stride, sync_bn=args.sync_bn, freeze_bn=args.freeze_bn)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model = model.cuda()
    model_u_time = time.time()
    model_load_time = model_u_time - model_s_time
    print("model load time is {}".format(model_load_time))
    composed_transforms = transforms.Compose([tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), tr.ToTensor()])
    L = 0
    N = 0
    for name in os.listdir(args.in_path):
        s_time = time.time()
        image = PIL.Image.open(args.in_path+"/"+name)
        target = image.convert("L")
        sample = {"image": image, "label": target}
        tensor_in = composed_transforms(sample)["image"].unsqueeze(0)
        model.eval()
        if args.cuda: tensor_in = tensor_in.cuda()

        with torch.no_grad():
            output = model(tensor_in)
        
        label = torch.max(output[:3], 1)[1].detach().squeeze().cpu().numpy()

        meter = meter_mask_info(label.astype(np.uint8), map_dict[osp.splitext(name)[0]])
        read = meter.reading_process()
        
        real = map_dict[osp.splitext(name)[0]]["scale_read"]
        loss = abs(real - read) / (real + 1e-8)
        label_names = ["{:.3f}".format(real),"{:.3f}".format(read),"{:.3f}".format(loss)]

        #alpha = 0.4
        #lbl_viz = (1 - alpha) * np.array(image, float) + alpha * (label_colormap()[label]).astype(float)
        #lbl_viz = np.clip(lbl_viz.round(), 0, 255).astype(np.uint8)
        lbl_viz = label2rgb(label=label, img=np.array(image), label_names=label_names, loc="rb", font_path=font_path)
        PIL.Image.fromarray(lbl_viz).save(osp.join(args.out_path,name))
        u_time = time.time()
        img_time = u_time-s_time
        print("image:{} time: {} real: {} read: {} loss: {} \n".format(name,img_time,real,read,loss))
        L += loss
        N += 1
    print("image save in in_path. avg_loss:{:.3f}".format(L/N))

if __name__ == "__main__":
    main()

# cd C:\Users\admin\Downloads\pytorch-deeplab-xception\
# conda activate pytorch
# python meter_infer.py --in_path _images_ --out_path _output_ --ckpt run\meter_seg_voc\deeplab-resnet\model_best2.pth.tar --backbone resnet
