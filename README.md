# pytorch-deeplab-xception

**Update on 2018/12/06. Provide model trained on VOC and SBD datasets.**  

**Update on 2018/11/24. Release newest version code, which fix some previous issues and also add support for new backbones and multi-gpu training. For previous code, please see in `previous` branch**  

### TODO
- [x] Support different backbones
- [x] Support VOC, SBD, Cityscapes and COCO datasets
- [x] Multi-GPU training



| Backbone  | train/eval os  |mIoU in val |Pretrained Model|
| :-------- | :------------: |:---------: |:--------------:|
| ResNet    | 16/16          | 78.43%     | [google drive](https://drive.google.com/open?id=1NwcwlWqA-0HqAPk3dSNNPipGMF0iS0Zu) |
| MobileNet | 16/16          | 70.81%     | [google drive](https://drive.google.com/open?id=1G9mWafUAj09P4KvGSRVzIsV_U5OqFLdt) |
| DRN       | 16/16          | 78.87%     | [google drive](https://drive.google.com/open?id=131gZN_dKEXO79NknIQazPJ-4UmRrZAfI) |



### Introduction
This is a PyTorch(0.4.1) implementation of [DeepLab-V3-Plus](https://arxiv.org/pdf/1802.02611). It
can use Modified Aligned Xception and ResNet as backbone. Currently, we train DeepLab V3 Plus
using Pascal VOC 2012, SBD and Cityscapes datasets.

![Results](doc/results.png)


### Installation
The code was tested with Anaconda and Python 3.6. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
    cd pytorch-deeplab-xception
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    pip install matplotlib pillow tensorboardX tqdm
    ```
### Training
Follow steps below to train your model:

0. Configure your dataset path in [mypath.py](https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/mypath.py).

1. Input arguments: (see full input arguments via python train.py --help):
    ```Shell
    usage: train.py [-h] [--backbone {resnet,xception,drn,mobilenet}]
                [--out-stride OUT_STRIDE] [--dataset {pascal,coco,cityscapes}]
                [--use-sbd] [--workers N] [--base-size BASE_SIZE]
                [--crop-size CROP_SIZE] [--sync-bn SYNC_BN]
                [--freeze-bn FREEZE_BN] [--loss-type {ce,focal}] [--epochs N]
                [--start_epoch N] [--batch-size N] [--test-batch-size N]
                [--use-balanced-weights] [--lr LR]
                [--lr-scheduler {poly,step,cos}] [--momentum M]
                [--weight-decay M] [--nesterov] [--no-cuda]
                [--gpu-ids GPU_IDS] [--seed S] [--resume RESUME]
                [--checkname CHECKNAME] [--ft] [--eval-interval EVAL_INTERVAL]
                [--no-val]

    eg.
    python train.py --backbone mobilenet --lr 0.005 --workers 1 --epochs 200 --batch-size 4 --gpu-ids 0 --checkname deeplab-mobilenet --dataset meter_seg_voc
    python train.py --backbone resnet --lr 0.005 --workers 1 --epochs 200 --batch-size 2 --gpu-ids 0 --checkname deeplab-resnet --dataset meter_seg_voc
    python train.py --backbone drn --lr 0.005 --workers 1 --epochs 200 --batch-size 4 --gpu-ids 0 --checkname deeplab-drn --dataset meter_seg_voc

    backbone_download_path:
    C:\Users\hkc/.cache\torch\hub\checkpoints\mobilenet_v2-6a65762b.pth

    ```

2. To train deeplabv3+ using Pascal VOC dataset and ResNet as backbone:
    ```Shell
    bash train_voc.sh
    ```
3. To train deeplabv3+ using COCO dataset and ResNet as backbone:
    ```Shell
    bash train_coco.sh
    ```    

### Tensorboard
You can visualize in real time the train and test losses, the weights and gradients, along with the model predictions with tensorboard:

`tensorboard --logdir=runs`

such as:
tensorboard --logdir=./run/meter_seg_voc/deeplab-resnet/experiment_0

visualize GPU：

watch -n 10 nvidia-smi

### predict

python predict.py --in-path your_file --out-path your_dst_file

such as:
python predict.py --in-path E:\sc\image_data\meter\meter_seg\images\val --ckpt run\meter_seg_voc\deeplab-resnet\model_best.pth.tar --backbone resnet

python predict.py --in-path /home/y/sc_dev/dilun/image_data/meter/meter_seg/images/val/ --ckpt run/meter_seg_voc/deeplab-resnet/model_best.pth.tar --backbone resnet


### Acknowledgement
[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)

[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

[drn](https://github.com/fyu/drn)
