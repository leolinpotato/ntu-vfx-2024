import numpy as np
import cv2
import math

from utils import *

def xy2hth(x, y, f):
    th = math.atan(x / f)
    h = y / math.sqrt(x**2 + f**2)
    return h, th

def hth2xy(th, h, f):
    x = math.tan(th) * f
    y = h * math.sqrt(x**2 + f**2)
    return x, y

def image_to_cylinder(image, f):
    y, x, _ = image.shape
    s = f
    x_max = int(s * math.atan((x / 2) / f)) * 2
    y_max = int(s * y / f)
    blank = np.zeros((y_max, x_max,3), dtype=np.uint8)
    '''
    for i in range(x):
        for j in range(y):
            i_ = i - int(x / 2)
            h, th = xy2hth(i_, j, f)
            h *= s
            th *= s
            h = int(h)
            th = int(th)
            th += int(x_max/2)
            blank[h][th] = image[j][i]
    '''
    for i in range(x_max):
        for j in range(y_max):
            from_x, from_y = hth2xy((i - x_max / 2)/s, (j - y_max / 2)/s, f)
            from_x += int(x / 2)
            from_y += int(y / 2)
            if from_x >= x or from_y >= y or from_x < 0 or from_y < 0:
                continue
            blank[j][i] = image[int(from_y)][int(from_x)]


    #image_show(blank)
    return blank


def images_to_cylinder(images, focal_lengths):
    cylinder_imgs = []
    i = 0
    for image, f in zip(images, focal_lengths):
        cylinder_imgs.append(image_to_cylinder(image, f))
        print("finish", i)
        i += 1
    return cylinder_imgs