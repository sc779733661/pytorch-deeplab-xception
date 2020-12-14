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
from PIL import Image
import PIL.ImageDraw
import PIL.ImageFont
import json
from torchvision import transforms
import matplotlib
import meter_mask_info as Mmi


def rectangle(src, aabb1, aabb2, fill=None, outline=None, width=0):
    if outline is not None:
        outline = tuple(outline)
    if fill is not None:
        fill = tuple(fill)

    dst = Image.fromarray(src)
    draw = PIL.ImageDraw.ImageDraw(dst)

    y1, x1 = aabb1
    y2, x2 = aabb2
    draw.rectangle(xy=(x1, y1, x2, y2), fill=fill, outline=outline, width=width )

    return np.array(dst)


def text_size(text, size, font_path=None):
    font = PIL.ImageFont.truetype(font=font_path, size=size)
    lines = text.splitlines()
    n_lines = len(lines)
    longest_line = max(lines, key=len)
    width, height = font.getsize(longest_line)
    return height * n_lines, width


def draw_text(src, yx, text, size, color=(0, 0, 0), font_path=None):
    dst = Image.fromarray(src)
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
        hsv = Image.fromarray(rgb, mode="RGB")
        hsv = hsv.convert("HSV")
        hsv = np.array(hsv)

        if isinstance(value, float):
            hsv[:, 1:, 2] = hsv[:, 1:, 2].astype(float) * value
        else:
            assert isinstance(value, int)
            hsv[:, 1:, 2] = value
        rgb = Image.fromarray(hsv, mode="HSV")
        rgb = rgb.convert("RGB")
        cmap = np.array(rgb).reshape(-1, 3)

    return cmap


def label2rgb(label, img=None, alpha=0.5, label_names=None, font_size=30,
              thresh_suppress=0, colormap=None,
              loc="centroid", font_path=None):
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
            gray = Image.fromarray(np.asarray(res[y, x], dtype=np.uint8).reshape(1, 1, 3)).convert("L")

            if np.array(gray).sum() > 170:
                color = (0, 0, 0)
            else:
                color = (255, 255, 255)
            
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


def load_json(path):
    with open(path, "r") as f:
        d = json.load(f)
    return d


def main():
    np.set_printoptions(threshold=np.inf)  # 控制输出的值的个数为threshold，其余以...代替,inf表示一个无限大的正数
    font_path = osp.join(osp.dirname(matplotlib.__file__), "mpl-data/fonts/ttf", "DejaVuSansMono.ttf")

    Loss_sum = 0
    Num_img = 0
    num_ng = 0
    time_sum = 0
    original_scale_num = 0
    model_score = 0
    map_dicts = load_json(r"json/mapping.json")
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument("--in_path", type=str, required=True, help="image to test")
    parser.add_argument("--out_path", type=str, required=True, help="mask image to save")
    parser.add_argument("--backbone", type=str, default="resnet", 
        choices=["resnet", "xception", "drn", "mobilenet"], help="backbone name (default: resnet)")
    parser.add_argument("--ckpt", type=str, default="deeplab-resnet.pth", help="saved model")
    parser.add_argument("--out_stride", type=int, default=16, help="network output stride (default: 8)")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--gpu_ids", type=str, default="0",
        help="use which gpu to train, must be a comma-separated list of integers only (default=0)")
    parser.add_argument("--crop_size", type=int, default=513, help="crop image size")
    parser.add_argument("--num_classes", type=int, default=3, help="crop image size")
    parser.add_argument("--save_reasultimg", type=bool, default=False, help="save reasult image")
    parser.add_argument("--sync_bn", type=bool, default=None, help="whether to use sync bn (default: auto)")
    parser.add_argument("--freeze_bn", type=bool, default=False,
        help="whether to freeze bn parameters (default: False)")
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
    model_load_time = model_u_time - model_s_time
    print("model load time is {}".format(model_load_time))
    composed_transforms = transforms.Compose([
        tr.Resize(crop_size=args.crop_size),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    for name in os.listdir(args.in_path):
        start_time = time.time()
        image = Image.open(os.path.join(args.in_path, name)).convert('RGB')
        target = image.convert("L")
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  # PIL-CV2
        # target = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2BGR)  # PIL-CV2
        sample = {"image": image, "label": target}
        tensor_in = composed_transforms(sample)["image"].unsqueeze(0)
        model.eval()
        if args.cuda:
            tensor_in = tensor_in.cuda()
        with torch.no_grad():
            output = model(tensor_in)
        end_model_time = time.time()
        # print(output.shape)

        label = torch.max(output[:3], 1)[1].detach().squeeze().cpu().numpy()
        map_dict = map_dicts[osp.splitext(name)[0]]
        meter = Mmi.Meter_mask_info(label.astype(np.uint8), map_dict)
        read, original_scale_num = meter.reading_process()
        real = map_dict["scale_read"]
        min_scale = map_dict["scale_max"] / (map_dict["scale_num"]-1)
        loss = abs(real - read) / (real + 1e-8)
        min_scale_loss = abs(real - read) / min_scale
        if original_scale_num == map_dict["scale_num"]:
            model_score += 1

        #alpha = 0.4
        #lbl_viz = (1 - alpha) * np.array(image, float) + alpha * (
        #           label_colormap()[label]).astype(float)
        #lbl_viz = np.clip(lbl_viz.round(), 0, 255).astype(np.uint8)
        if args.save_reasultimg:
            label_names = ["{:.3f}".format(real), "{:.3f}".format(read),
                       "{:.3f}".format(min_scale_loss)]
            # image = image.resize((args.crop_size, args.crop_size))
            image = cv2.resize(np.asarray(image), (args.crop_size, args.crop_size))
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            lbl_viz = label2rgb(label=label, img=np.array(image),
                                label_names=label_names, loc="rb",
                                font_path=font_path)
            Image.fromarray(lbl_viz).save(osp.join(args.out_path, name))
            print('image save in out_path.')

        end_time = time.time()
        modeltime = end_model_time-start_time
        img_time = end_time-start_time
        print("image:{} model time:{:.5f}s total time:{:.5f}s real value:{:.6f} read value:{:.6f} loss:{:.5f} min_scale_loss:{:.5f}\n".format(
            name, modeltime, img_time, real, read, loss, min_scale_loss))
        if min_scale_loss > 1:
            num_ng += 1
        Loss_sum += min_scale_loss
        Num_img += 1
        time_sum += img_time
    print("Model_score:{:.3f} ,Avg_loss:{:.3f} ,Accuracy(min_scale_loss<1):{:.3f}% ,Wrong_num:{} ,Total_num:{} ,Avg_time:{:.5f}".format(
        model_score/Num_img, Loss_sum/Num_img, (Num_img - num_ng)/Num_img*100, num_ng, Num_img, time_sum/Num_img))

if __name__ == "__main__":
    main()

# cd C:\Users\admin\Downloads\pytorch-deeplab-xception\
# conda activate pytorch
# python meter_infer.py --in_path _images_ --out_path _output_ --ckpt run\meter_seg_voc\deeplab-resnet\model_best.pth.tar --backbone resnet
# python meter_infer.py --in_path E:\sc\image_data\ttt\test --out_path E:\sc\image_data\ttt\resualt 
#                       --ckpt run\meter_seg_voc\deeplab-resnet\model_best.pth.tar --backbone resnet --save_reasultimg False
# python meter_infer.py --in_path /home/y/sc_dev/dilun/image_data/test_data/ --out_path /home/y/sc_dev/dilun/image_data/ttt/resualt/ 
#                       --ckpt run/meter_seg_voc/deeplab-resnet/model_best.pth.tar
