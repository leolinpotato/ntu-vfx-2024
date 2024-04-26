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
    return 0
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
    for x_ in range(5, int(board_size_x) + 5):
        for y_ in range(5, int(board_size_y) + 5):
            x = x_ + x_left - 5
            y = y_ + y_up - 5
            #A = np.identity(3)
            involve_img = []
            for i in range(len(images)):
                A_inv = np.linalg.inv(As[i])
                original_point = np.matmul(A_inv, np.array([x, y, 1]))
                #print("inv:", A_inv)
                #print("i point lrud:", i, original_point, images_lrud[i])
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
                board[int((y_ - y_shift(total_y_shift, total_x_shift, x_)) % board_size_y)][x_] = tmp
            
    image_show(board)
    return 0





    A = np.identity(3)
    for i in range(len(images)):
        for x in range(shape_x) :
            for y in range(shape_y) :
                
                new_point = np.matmul(A, np.array([x, y, 1]))
                new_x = new_point[0] - x_left + 5
                new_y = new_point[1] - y_up + 5
                if (board[int(new_y)][int(new_x)].all == 0) :
                    board[int(new_y)][int(new_x)] = np.uint8(images[i][y][x])
                else :
                    #print("yes")
                    if (images[i][y][x].all == 0):
                        continue
                    tmp = board[int(new_y)][int(new_x)].astype(float) / 2
                    board[int(new_y)][int(new_x)] = np.uint8(tmp)
                    board[int(new_y)][int(new_x)] += np.uint8(images[i][y][x] / 2)
                    
                #board[int(new_y)][int(new_x)] += np.uint8(images[i][y][x] / 2)
        if i != len(images) - 1 :
            A = np.matmul(A, best_shifts[i])
        image_show(board)
    image_show(board)
    '''
    for k in range(10):
        for x in range(1, 2999):
            for y in range(1, 2999):
                if (board[y][x].all == 0) :
                    tmp = board[y-1][x].astype(float) + board[y][x-1].astype(float) + board[y+1][x].astype(float) + board[y][x+1].astype(float)
                    tmp /= 4
                    #tmp /= 4
                    board[y][x] = np.uint8(tmp)
        image_show(board)
    '''
    return 0
    
    for x in range(shape_x) :
        for y in range(shape_y) :
            new_point = np.matmul(A, np.array([x, y, 1]))
            new_x = new_point[0] + 300
            new_y = new_point[1] + 2000

            board[int(new_y)][int(new_x)] += np.uint8(images[1][y][x] / 2)
    image_show(board)
    A = np.matmul(best_shifts[1], A)
    for x in range(shape_x) :
        for y in range(shape_y) :
            new_point = np.matmul(A, np.array([x, y, 1]))
            new_x = new_point[0] + 300
            new_y = new_point[1] + 300

            board[int(new_y)][int(new_x)] += np.uint8(images[2][y][x] / 2)
    
    image_show(board)
    return 0
    i = 0
    for img_mat in images_matching:
        for tmp in img_mat:
            matches, matches_id, j, best_inlier, best_shift = tmp
            if abs(j - i) != 1 or j < i:
                #print("pass", i, j)
                continue
            A = best_shift
            for x in range(shape_x) :
                for y in range(shape_y) :
                    new_point = np.matmul(A, np.array([x, y, 1]))
                    new_x = new_point[0] + 300
                    new_y = new_point[1] + 300
                    board[int(new_y)][int(new_x)] = images[j][y][x]
            image_show(board)
        i += 1