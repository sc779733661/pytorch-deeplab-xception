from numba import njit
import math
import numpy as np


@njit
def creat_line_image(meter_image, line_height, line_width,
                     circle_radius, circle_center):
    line_image = np.zeros((line_height, line_width), dtype=np.uint8)
    for row in range(line_height):
        for col in range(line_width):
            theta = math.pi * 2 / line_width * (col + 1)
            rho = circle_radius - row - 1
            x = int(circle_center[0] + rho * math.cos(theta) + 0.5)
            y = int(circle_center[1] - rho * math.sin(theta) + 0.5)
            line_image[row, col] = meter_image[x, y]
    return line_image


@njit
def convert_1d_data(meter_image, line_width, line_height):
    scale_data = np.zeros((line_width), dtype=np.uint8)
    pointer_data = np.zeros((line_width), dtype=np.uint8)
    for col in range(line_width):
        for row in range(line_height):
            if meter_image[row, col] == 1:
                pointer_data[col] += 1
            elif meter_image[row, col] == 2:
                scale_data[col] += 1
    return scale_data, pointer_data


@njit
def scale_mean_filtration(scale_data, line_width):
    mean_data = np.mean(scale_data)
    for col in range(line_width):
        if scale_data[col] < mean_data:
            scale_data[col] = 0
