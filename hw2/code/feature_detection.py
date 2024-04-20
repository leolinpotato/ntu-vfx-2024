import numpy as np
import cv2
import copy

from utils import *

def Harris_detection(image, k=0.04, threshold=0.01):
	# 1. Compute x and y derivatives of image
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	kernel = (5, 5)
	GI = cv2.GaussianBlur(gray_image, kernel, 0)
	Iy, Ix = np.gradient(GI)

	# 2. Compute products of deerivatives at every pixels
	Ixx = np.multiply(Ix, Ix)
	Iyy = np.multiply(Iy, Iy)
	Ixy = np.multiply(Ix, Iy)

	# 3. Compute the sums of the products of derivatives at each pixel
	kernel = (5, 5)
	Sxx = cv2.GaussianBlur(Ixx, kernel, 0)
	Syy = cv2.GaussianBlur(Iyy, kernel, 0)
	Sxy = cv2.GaussianBlur(Ixy, kernel, 0)

	# 4. Define the matrix at each pixel
	# 5. Compute the response of the detect at each pixel
	# -> Instead of doing this pixel wise, we implement in a map wise manner
	detM = Sxx*Syy - Sxy*Sxy
	traceM = Sxx + Syy
	R = detM - k*(traceM**2)

	# 6. Threshold on value of R; compute nonmax suppresion
	sigma = R.max()*threshold
	keypoints = []
	h, w = gray_image.shape
	for i in range(1, h - 1):
		for j in range(1, w - 1):
			pixel = R[i, j]
			neighbor = R[i-1:i+2, j-1:j+2]
			if (pixel > sigma) and ((pixel >= neighbor).all()):
				keypoints.append([i, j])
	
	# use red to highlight the feature points
	'''
	highlight_image = copy.deepcopy(image)
	for point in keypoints:
		x, y = point
		highlight_image[x-2:x+3, y-2:y+3] = [0, 0, 255]
	image_show(highlight_image)
	'''
	

	return keypoints