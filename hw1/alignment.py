import numpy as np
import math
from PIL import Image
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


image_1 = Image.open('DSCF4452.jpg')
print(image_1.size)
image_2 = Image.open('DSCF4453.jpg')
image_3 = Image.open('DSCF4454.jpg')
image_4 = Image.open('DSCF4455.jpg')
image_5 = Image.open('DSCF4456.jpg')
image_6 = Image.open('DSCF4458.jpg')
image_7 = Image.open('DSCF4459.jpg')
image_8 = Image.open('DSCF4460.jpg')
image_9 = Image.open('DSCF4461.jpg')
image_10 = Image.open('DSCF4462.jpg')
print("1, 2:", alignment(image_1, image_2, 6240, 4160, 7))
print("1, 3:", alignment(image_1, image_3, 6240, 4160, 7))
print("1, 4:", alignment(image_1, image_4, 6240, 4160, 7))
print("1, 5:", alignment(image_1, image_5, 6240, 4160, 7))
print("1, 6:", alignment(image_1, image_6, 6240, 4160, 7))
print("1, 7:", alignment(image_1, image_7, 6240, 4160, 7))
print("1, 8:", alignment(image_1, image_8, 6240, 4160, 7))
print("1, 9:", alignment(image_1, image_9, 6240, 4160, 7))
print("1, 10:", alignment(image_1, image_10, 6240, 4160, 7))







       
        
        

    
