import numpy as np
import math
import cv2
from jit_functions.jit import *


class Meter_mask_info():
    def __init__(self, _mask, _dict=None):
        H, W = _mask.shape
        self.circle_center = [H/2, W/2]
        self.circle_radius = min(H, W) / 2
        self.line_height = int(self.circle_radius*0.6)
        self.line_width = int(self.circle_radius*2*math.pi)
        self.kernel_size = 3  # magic num
        self.label_maps = _mask
        self.infos_dict = _dict

    def reading_process(self):
        # Normalizing and corrosion semantic map
        norm_images = self.norm_erode_image(self.label_maps)
        # Convert the circular meter into rectangular meter
        line_images = creat_line_image(norm_images,
                                       self.line_height,
                                       self.line_width,
                                       self.circle_radius,
                                       self.circle_center)
        # Convert the 2d meter into 1d meter
        scale_data, pointer_data = convert_1d_data(
                                        line_images,
                                        self.line_width,
                                        self.line_height)
        # Fliter scale data whose value is lower than the mean value
        scale_mean_filtration(scale_data, self.line_width)
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

    '''
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
    '''

    '''
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
    '''

    '''
    def scale_mean_filtration(self, scale_data):
        mean_data = np.mean(scale_data)
        for col in range(self.line_width):
            if scale_data[col] < mean_data:
                scale_data[col] = 0
    '''

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
                    print('fix merge scale')
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
        print(result)
        return result

    def fix_initial_scale(self, result):
        if (self.infos_dict["scale_num"] - result['scale_num']) > 2:  # 如果检测刻度出错，直接用ratio
            result['ratio'] = result['ratio']
            print('ratio')
            return result
        # else:
        #     result['ratio'] = result['ratio_add']
        #     print('ratio = ratio_add')
        if ((result['scale_width_first'] > result['scale_width_mean']*1.5) and (
            result['scale_num'] < self.infos_dict["scale_num"])):  # 如果第一个刻度过宽，说明第一二个刻度合并
            print('add initial scale.')
            result['scale_num'] = result['scale_num'] + 1
            result['scales'] = result['scales'] + 1
            result['ratio'] = result['ratio'] + result['onescalerange_ratio']*0.5  # 角度要加上0.5个刻度角度
            print('ratio + onescalerange_ratio*0.5')
        else:
            if (self.infos_dict["scale_num"] - result['scale_num']) == 1:  # 没有过宽，但确实合并了
                print('add 1')
                result['scale_num'] = result['scale_num'] + 1
                result['scales'] = result['scales'] + 1
                result['ratio'] = result['ratio'] + result['onescalerange_ratio']
                print('ratio + onescalerange_ratio')
                return result
        # 如果原表缺少第一个刻度:2种情况：1. 25，26类型，如果是单数且//5%2==1； 2. 30，31类型，如果是双数且//5%2==0
        if (self.infos_dict["scale_num"]%2 == 1) and (self.infos_dict["scale_num"]//5%2 == 1):  
            print('lach 1 scale')
            result['scales'] = result['scales'] + 1
            result['ratio'] = result['ratio'] + result['onescalerange_ratio']
            print('ratio + onescalerange_ratio')
        elif (self.infos_dict["scale_num"]%2 == 0) and (self.infos_dict["scale_num"]//5%2 == 0):
            print('lach 1 scale')
            result['scales'] = result['scales'] + 1
            result['ratio'] = result['ratio'] + result['onescalerange_ratio']
            print('ratio + onescalerange_ratio')
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
