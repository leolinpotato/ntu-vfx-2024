import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
import os
import random

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

def recover_response_curve(images, B, w):
	# random sample n pixels
	n = 50
	all_points = [(i, j) for i in range(len(images[0])) for j in range(len(images[0][0]))]
	sampled_points = random.sample(all_points, n)

	Z = np.zeros((3, n, len(images))).astype(int)
	for channel in range(3):
		for i in range(n):
			r, c = sampled_points[i]
			for j in range(len(images)):
				Z[channel,i,j] = images[j,r,c,channel]

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

def main():
	images, exposure_time = read_images("image")
	B = np.log(exposure_time)
	w = [min(i, 256-i)/128 for i in range(256)]

	g = recover_response_curve(images, B, w)
	E = recover_radiance_map(images, B, g, w)
	output_path = f"result/hdr.png"
	cv2.imwrite(output_path, E)
	
main()
