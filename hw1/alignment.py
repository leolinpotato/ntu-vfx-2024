import numpy as np
import math
from PIL import Image, ImageOps
import cv2
from os.path import splitext
import os, sys

def img_shift(img, n, m, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    img2 = cv2.warpAffine(img, M, (n, m))
    return img2

def compare(image_a, image_b, n, m, x, y) :
    image_c = img_shift(image_b, n, m, x, y)
    image_a_L = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    #print(image_a_L)
    image_c_L = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)
    MTB_a = img2MTB(image_a_L)
    MTB_c = img2MTB(image_c_L)
    #print(MTB_a)
    mask = mask_gen(image_a_L, image_c_L)
    
    xor_result = np.logical_xor(MTB_a, MTB_c)
    cnt = 0
    xor_result = np.logical_and(xor_result, mask)
    cnt = np.sum(xor_result)
    return cnt

def img2MTB(img):
    mid = np.mean(img)
    img2 = cv2.inRange(img, mid, 256)
    return img2

def mask_gen(a, b) :
    mid_a = np.mean(a)
    mid_b = np.mean(b)
    mask_a = cv2.inRange(a, mid_a - 4, mid_a + 4)
    mask_b = cv2.inRange(b, mid_b - 4, mid_b + 4)
    or_result = np.logical_or(mask_a, mask_b)
    or_result = np.logical_not(or_result)
    return or_result

def alignment(image_a, image_b, n, m, k) :
    #print(k)
    if (k > 7) :
        k = 7
    if k <= 0 :
        return (0, 0)
    new_n = math.floor(n / 2)
    new_m = math.floor(m / 2)
    new_image_a = cv2.resize(image_a, (new_n, new_m))
    new_image_b = cv2.resize(image_b, (new_n, new_m))
    pre_shift = alignment(new_image_a, new_image_b, new_n, new_m, k - 1)
    pre_shift = (pre_shift[0] * 2, pre_shift[1] * 2)
    #print(k)
    '''
    image_a_L = image_a.convert('L')
    image_b_L = image_b.convert('L')
    #print(image_a)
    array_a = np.array(image_a_L)
    #print(array_a.shape)
    array_b = np.array(image_b_L)
    #print(array_a)
    mask = np.full((m, n), True)
    #print(mask.shape)
    for c, c2 in zip(array_a, mask) :
        for p, p2 in zip(c, c2) :
            if 125 < p and p < 130 :
                p2 = False
            if p > 127 :
                p = True
            else :
                p = False 
            

    for c, c2 in zip(array_b, mask) :
        for p, p2 in zip(c, c2) :
            if 125 < p and p < 130 :
                p2 = False
            if p > 127 :
                p = True
            else :
                p = False 
    '''
    '''
    image_a_L = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    print(image_a_L)
    image_b_L = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    MTB_a = img2MTB(image_a_L)
    MTB_b = img2MTB(image_b_L)
    print(MTB_a)
    mask = mask_gen(image_a_L, image_b_L)
    '''
    min_offset = (0, 0)
    min_loss = n * m
    for i in range(pre_shift[0] - 1, pre_shift[0] + 2) :
        for j in range(pre_shift[1] - 1, pre_shift[1] + 2) :
            current_loss = compare(image_a, image_b, n, m, i, j)
            if current_loss < min_loss :
                min_offset = (i, j)
                min_loss = current_loss
    
    return min_offset



def padding(img, n, m, x, y):
    #cv2.rectangle(image, start_point, end_point, color, thickness) 
    #img2 = cv2.rectangle(img, (0, 0), (n - 400, m - 400), (0, 0, 0) , -1)
    img2 = cv2.rectangle(img, (0, 0), (n - 1, y), (0, 0, 0) , -1)
    img2 = cv2.rectangle(img2, (0, 0), (x, m - 1), (0, 0, 0) , -1)
    img2 = cv2.rectangle(img2, (n - x - 1, 0), (n - 1, m - 1), (0, 0, 0) , -1)
    img2 = cv2.rectangle(img2, (0, m - 1 - y), (n - 1, m - 1), (0, 0, 0) , -1)
    return img2


def main():
    dir_name = 'image/'
    os.makedirs(dir_name + 'align', exist_ok = True)
    image_names = ['DSCF4452.jpg', 'DSCF4453.jpg', 'DSCF4454.jpg',  'DSCF4455.jpg', 'DSCF4456.jpg', 'DSCF4458.jpg', 'DSCF4459.jpg', 'DSCF4460.jpg', 'DSCF4461.jpg', 'DSCF4462.jpg']
    images = []
    for name in image_names:
        tmpimg = cv2.imread(dir_name + name)
        images.append(tmpimg)
    i = 0
    shifts = [(0, 0)]
    for img in images:
        i += 1
        if (i == 5) :
            continue
        shift = alignment(images[4], img, 6240, 4160, 7)
        print('5, ', i, ': ', shift)
        shifts.append(shift)
    
    max_shift = (0, 0)
    for shift in shifts:
        if abs(shift[0]) > max_shift[0] :
            max_shift = (abs(shift[0]), max_shift[1])
        if abs(shift[1]) > max_shift[1] :
            max_shift = (max_shift[0], abs(shift[1]))
    print(max_shift)
    
    i = 0
    shifted_img = []
    for img, shift in zip(images, shifts):
        shifted_img.append(img_shift(img, 6240, 4160, shift[0], shift[1]))
        shifted_img[i] = padding(shifted_img[i], 6240, 4160, max_shift[0], max_shift[1])
        i += 1
    i = 0
    for img in shifted_img:
        name = splitext(image_names[i])[0] + '_align' + '.jpg'
        cv2.imwrite(dir_name + 'align/' + name, img)
        i += 1
    

'''
def main2():
    dir_name = 'image/'
    image_names = ['DSCF4452.jpg', 'DSCF4453.jpg', 'DSCF4454.jpg',  'DSCF4455.jpg', 'DSCF4456.jpg', 'DSCF4458.jpg', 'DSCF4459.jpg', 'DSCF4460.jpg', 'DSCF4461.jpg', 'DSCF4462.jpg']
    images = []
    for name in image_names:
        tmpimg = Image.open(dir_name + name)
        images.append(tmpimg)
    i = 0
    shifts = [(0, 0), (3, 3), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    max_shift = (0, 0)
    for shift in shifts:
        if shift[0] > max_shift[0] :
            max_shift = (shift[0], max_shift[1])
        if shift[1] > max_shift[1] :
            max_shift = (max_shift[0], shift[1])
    i = 0
    for img, shift in zip(images, shifts):
        img = img_cut(img, max_shift[0] - shift[0], max_shift[1] - shift[1], shift[0], shift[1])
    for img in images:
        name = splitext(image_names[i])[0] + '_' + str(i) + '.jpg'
        img.save(dir_name + name)
        i += 1
'''
main()












       
        
        

    
