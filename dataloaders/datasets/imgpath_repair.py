import os

def changepath(path,topath):
    file = open(path, 'r')
    lines = []
    for i in file:
        lines.append(i)  #逐行将文本存入列表lines中
    file.close()

    new = []
    for line in lines:  # 逐行遍历
        line = line.replace('images/train/','')
        line = line.replace('images/val/','')
        p = 0  # 定义计数指针
        for bit in line:  # 对每行进行逐个字遍历
            if bit == '.':   # 遇到0时进行处理（我们可以修改成我们想要的定位）
                line = line[:p]
                break
            else:
                p = p + 1  # 如果bit不是空格，指针加一
        new.append(line)
    
    file_write = open(topath, 'w')
    for var in new:
        file_write.writelines(var)
        file_write.writelines('\n')
    file_write.close()


if __name__ == '__main__':
    changepath('E:/sc/image_data/meter/meter_seg_voc/ImageSet/Segmentation/val_yuan.txt'
               , 'E:/sc/image_data/meter/meter_seg_voc/ImageSet/Segmentation/val.txt')
