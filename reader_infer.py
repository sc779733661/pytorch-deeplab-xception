# 测试检测指针度数
import cv2
import os
import numpy as np
import glob
import random


if __name__ == '__main__':
    src_img = cv2.imread('E:\\sc\\image_data\\meter\\meter_seg\\images\\val\\6_mask.png')
    print(src_img.shape)
    # magic number
    kernelsize = 5

    # 腐蚀
    kernel = np.ones((kernelsize,kernelsize), np.uint8)
    erosion_img = cv2.erode(src_img, kernel, iterations=1)

    # 展开环形表盘
    onec_img = cv2.cvtColor(erosion_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(onec_img, 20,255,0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]

    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    circle_img = cv2.circle(erosion_img, center, radius, (0,255,2),2)


    cv2.imshow('image',thresh)
    cv2.waitKey(0)