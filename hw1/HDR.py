import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
import os
import random
# from JBF import Joint_bilateral_filter
import natsort

def read_images(dir_path):
	images = []
	file_list = os.listdir(dir_path)

	image_files = [f for f in file_list if f.lower().endswith(('.jpg'))]
    
	# Initialize an empty list to store images
	images = []
	exposure_time = []
	# Read each image file and append it to the images list
	for image_file in natsort.natsorted(image_files):
		image_path = os.path.join(dir_path, image_file)
		image = cv2.imread(image_path)
		# Convert BGR image to RGB (cv2 uses BGR by default)
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		images.append(image)
		exposure_time.append(np.float32(get_exif_data(image_path)))

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

def recover_response_curve(images, B, w):
	# random sample n pixels
	'''
	n = 1000
	all_points = [(i, j) for i in range(len(images[0])) for j in range(len(images[0][0]))]
	sampled_points = random.sample(all_points, n)

	Z = np.zeros((3, n, len(images))).astype(int)
	for channel in range(3):
		for i in range(n):
			r, c = sampled_points[i]
			for j in range(len(images)):
				Z[channel,i,j] = images[j,r,c,channel]
	'''
	# down sample
	width, height = 60, 40
	down_sample = []
	for image in images:
		down_sample.append(cv2.resize(image, (width, height)))
	down_sample = np.array(down_sample)
	Z = np.zeros((3, width*height//4, len(images))).astype(int)
	for channel in range(3):
		for i in range(width*height//4):
			r, c = i//(width//2)+height//4, i%(width//2)+width//4
			for j in range(len(images)):
				Z[channel,i,j] = down_sample[j,r,c,channel]

	l = 0.01

	g = np.zeros((3, 256))
	for i in range(3):
		g[i] = gsolve(Z[i], B, l, w)
	return g

def gsolve(Z, B, l, w):
	'''
	Z[i, j]: value of ith pixel in image j
	B      : exposure time, ln delta t
	l      : lambda, parameter to regularize, make g smoother
	w      : weight function
	'''
	n, p = Z.shape
	c = 256
	A = np.zeros((n*p+(c - 2)+1, c+n))
	b = np.zeros((A.shape[0], 1))

	# fill A
	k = 0  # row in A
	for i in range(n):
		for j in range(p):
			wij = w[Z[i, j]]
			A[k, Z[i, j]] = wij
			A[k, c+i] = -wij
			b[k, 0] = wij*B[j]
			k += 1

	A[k, 127] = 1
	k += 1

	for i in range(c - 2):
		A[k, i] = l*w[i]
		A[k, i+1] = -2*l*w[i]
		A[k, i+2] = l*w[i]
		k += 1

	x = np.linalg.lstsq(A, b, rcond=None)[0]  # return the least-squaure solution

	g = x[:c].reshape(-1)
	lE = x[c:].reshape(-1)

	return g

def recover_radiance_map(images, B, g, w):
	p = len(images)
	height, width, _ = images[0].shape
	E = np.zeros((height, width, 3))
	# for each pixel in an image
	for channel in range(3):
		for r in range(height):
			for c in range(width):
				lE = 0
				w_sum = 0
				for j in range(p):
					zij = images[j,r,c,channel]
					lE += w[zij]*(g[channel,zij]-B[j])
					w_sum += w[zij]
				if w_sum != 0:
					lE /= w_sum
				E[r,c,channel] = lE
	E = np.exp(E)
	return E

def reshape_images(images, ratio=1):
	for i in range(len(images)):
		h, w, _ = images[i].shape
		images[i] = cv2.resize(images[i], (w//ratio, h//ratio))
	images = np.array(images)
	return images

def image_show(image):
	cv2.imshow("image", image)
	cv2.waitKey(0)

def separate_intensity_color(image):
	yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
	Y, Cr, Cb = cv2.split(yuv_image)
	return Y, Cr, Cb

def global_operator(lw, a=0.7, l_white=2.5):
	delta = 1e-10
	# B, G, R = 0.114, 0.587, 0.299
	B, G, R = 0.06, 0.67, 0.27
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
	

def main():
	_, exposure_time = read_images("image")
	images, _ = read_images("image/align")
	images = reshape_images(images, 2)

	# reconstruct HDR image
	B = np.log(exposure_time)
	w = [min(i, 256-i)/128 for i in range(256)]

	g = recover_response_curve(images, B, w)
	E = recover_radiance_map(images, B, g, w)

	# tone-mapping to LDR image
	global_output, lm = global_operator(E)
	local_output = local_operator(lm)
	cv2.imwrite(f"result/global_tone_mapping_align.png", global_output)
	cv2.imwrite(f"result/local_tone_mapping_align.png", local_output)
	
	# Y, Cr, Cb = separate_intensity_color(E)
	# Y = np.clip(Y, 0, 255).astype(np.uint8)
	# image_YCbCr_adjusted = cv2.merge((Y, Cb, Cr))
	# image_RGB_adjusted = cv2.cvtColor(image_YCbCr_adjusted, cv2.COLOR_YCrCb2BGR)
	# image_show(image_RGB_adjusted)

	# sigma_s, sigma_r = 1, 0.05
	# JBF = Joint_bilateral_filter(sigma_s, sigma_r)
	# bf_out = JBF.joint_bilateral_filter(Y*255, Y*255).astype(np.uint8)
	# image_show(Y*255 - bf_out)
	# print(bf_out)
	# print(np.array(Y).mean())
	# print(Y*255 - bf_out)
	# cv2.imwrite(f"result/jbf.png", Y*255 - bf_out)

	
main()
