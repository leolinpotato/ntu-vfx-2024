import numpy as np
import cv2
import os
import natsort
import argparse

from utils import *
from image_matching import *


def in_range(x, y, lrud):
    x_left = lrud[0]
    x_right = lrud[1]
    y_up = lrud[2]
    y_down = lrud[3]
    if x > x_left and x < x_right and y > y_up and y < y_down:
        return True
    return False

def w(left, right, x):
    mid = (left + right) / 2
    if (x > mid):
        return right - x
    return x - left

def y_shift(total_y_shift, total_x_shift, x):
    return int(x * total_y_shift / total_x_shift)

def blending(images):
    
    images.append(images[0])
    images_matching = image_matching(images, cheat=True)
    shape_y, shape_x, _ = images[0].shape
    
    A = np.identity(3)
    i = 0
    #print("len: ", len(images_matching))
    best_shifts = []
    for img_mat in images_matching:
        for tmp in img_mat:
            matches, matches_id, j, best_inlier, best_shift = tmp
            if (j == i + 1):
                best_shifts.append(best_shift)
                #print(best_shift)
        i += 1
    leftup_point = np.array([0, 0, 1])
    leftdown_point = np.array([0, shape_y-1, 1])
    rightup_point = np.array([shape_x-1, 0, 1])
    rightdown_point = np.array([shape_x-1, shape_y-1, 1])
    x_left = 0
    x_right = shape_x-1
    y_up = 0
    y_down = shape_y-1
    images_lrud = [(0, shape_x-1, 0, shape_y-1)]
    As = [A]
    for shift in best_shifts:
        A = np.matmul(A, shift)
        As.append(A)
        new_leftup = np.matmul(A, leftup_point)
        new_leftdown = np.matmul(A, leftdown_point)
        new_rightup = np.matmul(A, rightup_point)
        new_rightdown = np.matmul(A, rightdown_point)
        images_lrud.append((new_leftup[0], new_rightdown[0], new_rightup[1], new_leftdown[1]))
        if new_leftup[0] < x_left :
            x_left = new_leftup[0]
        if new_leftdown[0] < x_left :
            x_left = new_leftdown[0]
        
        if new_leftup[1] < y_up :
            y_up = new_leftup[1]
        if new_rightup[1] < y_up :
            y_up = new_rightup[1]
        
        if new_rightup[0] > x_right :
            x_right = new_rightup[0]
        if new_rightdown[0] > x_right :
            x_right = new_leftdown[0]
        if new_leftdown[1] > y_down :
            y_up = new_leftup[1]
        if new_rightdown[1] > y_down :
            y_up = new_rightup[1]
    board_size_x = x_right - x_left
    board_size_y = y_down - y_up
    board = np.zeros((int(board_size_y) + 10, int(board_size_x) + 10,3), dtype = np.uint8)
    same_point_dif_plase = np.matmul(As[-1], np.array([1, 1, 1]))
    total_y_shift = 1 - same_point_dif_plase[1]
    total_x_shift = 1 - same_point_dif_plase[0]
    A_invs = []
    for A in As:
        A_invs.append(np.linalg.inv(A))
    for x_ in range(5, int(board_size_x) + 5):
        for y_ in range(5, int(board_size_y) + 5):
            x = x_ + x_left - 5
            y = y_ + y_up - 5
            #A = np.identity(3)
            involve_img = []
            for i in range(len(images)):
                A_inv = A_invs[i]
                original_point = np.matmul(A_inv, np.array([x, y, 1]))
                if in_range(original_point[0], original_point[1], (0, shape_x-1, 0, shape_y-1)):
                    involve_img.append(i)
            tmp = np.zeros(3)
            weight_sum = 0
            for i in involve_img:
                A_inv = np.linalg.inv(As[i])
                original_point = np.matmul(A_inv, np.array([x, y, 1]))
                new_x = int(original_point[0])
                new_y = int(original_point[1])
                tmp += images[i][new_y][new_x].astype(float) * w(0, shape_x, new_x)
                weight_sum += w(0, shape_x, new_x)
            if (weight_sum != 0):
                tmp /= weight_sum
                tmp = np.uint8(tmp)
                board[int((y_ - y_shift(total_y_shift, total_x_shift, x_)) % (board_size_y + 5))][x_] = tmp
            
    image_show(board)
    cv2.imwrite('output.jpg', board)
    return board