import numpy as np
import math
from PIL import Image
from os.path import splitext

def compare(a, b, mask, n, m, offset_x, offset_y) :
    c = np.roll(b, (offset_x, offset_y), axis = (0, 1))
    xor_result = np.logical_xor(a, c)
    cnt = 0
    xor_result = np.logical_and(xor_result, mask)
    '''
    for i in range(offset_x, n - 1):
        for j in range(offset_y, m - 1):
            if xor_result[i][j] == False :
                cnt += 1
    '''
    cnt = np.size(xor_result) - np.count_nonzero(xor_result)
    return cnt

def alignment(image_a, image_b, n, m, k) :
    #print(k)
    if (k > 7) :
        k = 7
    if k <= 0 :
        return (0, 0)
    new_n = math.floor(n / 2)
    new_m = math.floor(m / 2)
    new_image_a = image_a.resize((new_n, new_m))
    new_image_b = image_b.resize((new_n, new_m))
    pre_offset = alignment(new_image_a, new_image_b, new_n, new_m, k - 1)
    pre_offset = (pre_offset[0] * 2, pre_offset[0] * 2)
    #print(k)
    
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
    mask[0:2, 0:m] = False
    mask[n-2:n, 0:m] = False
    mask[0:n, 0:2] = False
    mask[0:n, m-2:m] = False
    min_offset = (0, 0)
    min_loss = n * m
    for i in range(pre_offset[0], pre_offset[0] + 2) :
        for j in range(pre_offset[1], pre_offset[1] + 2) :
            current_loss = compare(array_a, array_b, mask, n, m, i, j)
            if current_loss < min_loss :
                min_offset = (i, j)
                min_loss = current_loss
    
    return min_offset

def img_cut(img, x, y, z, w):
    img2 = img.crop((x, y, 6240 - z, 4167 - w))
    return img2

def main():
    dir_name = 'image/'
    image_names = ['DSCF4452.jpg', 'DSCF4453.jpg', 'DSCF4454.jpg',  'DSCF4455.jpg', 'DSCF4456.jpg', 'DSCF4458.jpg', 'DSCF4459.jpg', 'DSCF4460.jpg', 'DSCF4461.jpg', 'DSCF4462.jpg']
    images = []
    for name in image_names:
        tmpimg = Image.open(dir_name + name)
        images.append(tmpimg)
    i = 0
    shifts = [(0, 0)]
    for img in images:
        i += 1
        if (i == 1) :
            continue
        shift = alignment(images[0], img, 6240, 4160, 7)
        print('1, ', i, ': ', shift)
        shifts.append(shift)
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
main2()












       
        
        

    
