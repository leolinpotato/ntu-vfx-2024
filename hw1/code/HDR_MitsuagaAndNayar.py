import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
import os
import random
import math
import matplotlib.pyplot as plt
from scipy import optimize
#from HDR import local_operator, global_operator


pow_Z = [[pow(i, j) for j in range(0, 11)] for i in range(0, 256)]
def read_images(dir_path):
    images = []
    file_list = os.listdir(dir_path)

    image_files = [f for f in file_list if f.lower().endswith(('.jpg'))]

    # Initialize an empty list to store images
    images = []
    exposure_time = []
    # Read each image file and append it to the images list
    for image_file in image_files:
        image_path = os.path.join(dir_path, image_file)
        image = cv2.imread(image_path)
        # Convert BGR image to RGB (cv2 uses BGR by default)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        exposure_time.append(np.float32(get_exif_data(image_path)))
    images = np.array(images)
    exposure_time = np.array(exposure_time)
    return images, exposure_time

def get_exif_data(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Get the EXIF data
        exif_data = img._getexif()
        if exif_data is not None:
            # Iterate over all EXIF tags
            for tag, value in exif_data.items():
                # Decode the tag
                tag_name = TAGS.get(tag, tag)
                # Check if the tag is 'ExposureTime'
                if tag_name == 'ExposureTime':
                    # Return the exposure time
                    return value
'''
def reshape_images(images, ratio=1):
	for i in range(len(images)):
		h, w, _ = images[i].shape
		images[i] = cv2.resize(images[i], (w//ratio, h//ratio))
	images = np.array(images)
	return images
'''

def f(c, M, Z):
    a = 0
    for m in range(0, M + 1):
        a += c[m] * pow_Z[int(Z)][m]
        #print(a)
    return a

def f_d(c, M, Z):
    a = 0
    for m in range(1, M + 1):
        a += c[m] * pow_Z[int(Z)][m - 1] * m
        #print(a)
    return a

def calculate_R(j, c, Z, P, M):
    result = 0
    for i in range(0, P):
        result += (f(c, M, Z[i][j]) / f(c, M, Z[i][j+1]))
    #print("result:", result)
    result = result / P
    return result

def calculate_c(N, P, Z, M, R, Imax):
    A = np.zeros((M, M))
    b = np.zeros(M)
    for x in range(0, M):
        for y in range(0, M + 1):
            if (y == M):
                b[x] = 0
                for i in range(0, N):
                    for j in range(0, P - 1):
                        b[x] += (pow_Z[Z[i][j]][y] - R[j] * pow_Z[Z[i][j+1]][y]) * (pow_Z[Z[i][j]][x] - R[j] * pow_Z[Z[i][j+1]][x]) 
                b[x] *= (-1)
                for y2 in range(0, M):
                    A[x][y2] += b[x]
                continue
            A[x][y] = 0
            for i in range(0, N):
                for j in range(0, P - 1):
                    A[x][y] += (pow_Z[Z[i][j]][y] - R[j] * pow_Z[Z[i][j+1]][y]) * (pow_Z[Z[i][j]][x] - R[j] * pow_Z[Z[i][j+1]][x]) 
    c = np.linalg.solve(A, b)
    err = np.matmul(A, c) - b
    cM = Imax - f(c, M - 1, 1)
    c = np.append(c, cM)
    return c
    
def pick(P, N, images):
    #n, m = imgs[0].shape
    '''
    small_img = []
    for i in range(P):
        small_img.append(cv2.resize(imgs[i], (30, 20)))
    '''
    '''
    Z = np.zeros((N, P))
    random.seed()
    for i in range(0, N):
        
        x = random.randint(0, n-1)
        y = random.randint(0, m-1)
        for j in range(0, P):
            Z[i][j] = imgs[j][x][y]
    Z = Z.astype('int64')
    return Z
    '''
    width, height = 30, 20
    down_sample = []
    for image in images:
        down_sample.append(cv2.resize(image, (width, height)))
    down_sample = np.array(down_sample)
    Z = np.zeros((N, P)).astype(int)
    for i in range(N):
        r, c = i//(width//2)+height//4, i%(width//2)+width//4
        for j in range(P):
            Z[i,j] = down_sample[j,r,c]
    return Z

def calculate_hdr(images, N, M, P, mode, w_x):
    n, m = images[0].shape
    c = np.zeros(M + 1)
    for i in range(0, M + 1):
        c[i] = 0.1
    Z = pick(P, N, images)
    f_pre = 0
    f_now = 0
    for Zi in Z:
        for Zij in Zi:
            f_now += f(c, M, Zij)
    R = np.zeros((P - 1))
    for i in range(0, P - 1):
        R[i] = 0.5
    zero = 0.0001
    cnt_rnd = 0
    Imax = 2
    while(abs(f_pre - f_now) >= zero and cnt_rnd < 1000):
        cnt_rnd += 1
        c = calculate_c(N, P, Z, M, R, Imax)
        
        for j in range(0, P - 1):
            R[j] = calculate_R(j, c, Z, P, M)
        
        f_pre = f_now
        f_now = 0
        
        for Zi in Z:
            for Zij in Zi:
                f_now += f(c, M, Zij)
        #break
    
    E = np.zeros((n, m))
    exposure_time = [0 for i in range(P)]
    sum_R = np.sum(R) + 1
    exposure_time[0] = 1 / sum_R
    for i in range(1, P):
        exposure_time[i] = exposure_time[i - 1] * R[i - 1]
    f_t = [f(c, M, i) for i in range(256)]
    f_d_t = [f_d(c, M, i) for i in range(256)]
    
    w = [(i) for i in range(256)]
    for i in range(256):
        w[i] = ((i - w_x))**2
    
    #w = [(f_t[i] / f_d_t[i]) for i in range(256)]
    #plt.plot(w, "green")
    for i in range(0, n):
        for j in range(0, m):
            E[i][j] = 0
            w_sum = 0
            for p in range(0, P):
                Zij = images[p][i][j] 
                I1 = f_t[Zij]
                I2 = I1 / exposure_time[p]
                E[i][j] += I2 * w[Zij]
                w_sum += w[Zij]
            E[i][j] /= w_sum
    #E = E + abs(np.min(E)) + 1
    return E

def global_operator(lw, a=0.7, l_white=2.5):
    delta = 1e-10
    # B, G, R = 0.114, 0.587, 0.299
    B, G, R = 0.06, 0.67, 0.27
    #B, G, R = 0.33, 0.33, 0.33
    lw_bar = np.exp(np.mean(np.log(delta+lw[:,:,0])*B + np.log(delta+lw[:,:,1])*G + np.log(delta+lw[:,:,2])*R))
    lm = lw*a/lw_bar
    # print(np.array(lm).max())
    ld = lm*(1+lm/l_white**2)/(1+lm)
    output = np.clip(ld*255, 0, 255).astype(np.uint8)
    return output, lm

def local_operator(lm, a=0.7):
	# the parameter setting is based on the paper "Erik Reinhard, Michael Stark, Peter Shirley, Jim Ferwerda, Photographics Tone Reproduction for Digital Images, SIGGRAPH 2002."
	phi = 8
	threshold = 0.05
	l_blur = []
	scale = []
	pixel = 1.0
	for i in range(8):
		l_blur.append(cv2.GaussianBlur(lm, (int(2*np.rint(pixel)+1), int(2*np.rint(pixel)+1)), 0, 0))
		scale.append(int(2*np.rint(pixel)+1))
		pixel *= 1.6
	# find Smax
	Smax = 0
	for s in range(len(l_blur)-1):
		V = (l_blur[s]-l_blur[s+1])/((2**phi)*a/(scale[s]**2)+l_blur[s])
		if np.all(abs(V) < threshold):
			Smax = s
	ld = lm/(1+l_blur[Smax])
	output = np.clip(ld*255, 0, 255).astype(np.uint8)
	return output


def norm_3(vec):
    x, y, z = vec
    return math.sqrt(x**2 + y**2 + z**2)

def error_func(p, I, M):
    Mr, Mg, Mb = M
    Ir, Ig, Ib = I
    pr, pg, pb = p
    Ic = np.array([Ir * pr, Ig * pg, Ib * pb])
    Ic_M = Ic / norm_3(Ic) - M / norm_3(M)
    return norm_3(Ic_M)


def MitsuagaAndNayar_HDR(images, N, M, w_x):
    
    #N = 100
    P, _1, _2, _3 = images.shape
    #M = 5
    img_b = [0 for i in range(P)]
    img_g = [0 for i in range(P)]
    img_r = [0 for i in range(P)]
    small_img = [0 for i in range(P)]
    for i in range(P):
        #small_img[i] = cv2.resize(images[i], (int(m / 4), int(n / 4)))
        img_b[i], img_g[i], img_r[i] = cv2.split(images[i])
    
    #w_x = 210
    E_r = calculate_hdr(img_r, N, M, P, "red", w_x)
    E_b = calculate_hdr(img_b, N, M, P, "blue", w_x)
    E_g = calculate_hdr(img_g, N, M, P, "green", w_x)

    E_b_mean = np.mean(E_b)
    E_g_mean = np.mean(E_g)
    E_r_mean = np.mean(E_r)
    M_b_mean = np.mean(img_b[4])
    M_g_mean = np.mean(img_g[4])
    M_r_mean = np.mean(img_r[4])

    I = np.array([E_r_mean, E_g_mean, E_b_mean])
    M = np.array([M_r_mean, M_g_mean, M_b_mean])
    x0 = np.array([1, 1, 1])

    fit_res = optimize.least_squares(error_func, x0, args=(I, M))
    result_x = fit_res['x']
    pr, pg, pb = result_x
    E_r = E_r * pr / pb
    E_g = E_g * pg / pb
    E_b = E_b 

    E = cv2.merge((E_b, E_g, E_r))
    '''
    global_output, lm = global_operator(E)
    cv2.imwrite(f"result/global_tone_mapping_align_MAN_" + str(w_x) + ".jpg", global_output)
    #print(global_output)
    local_output = local_operator(lm)
    cv2.imwrite(f"result/local_tone_mapping_align_MAN_" + str(w_x) + ".png", local_output)
    '''
    return E
