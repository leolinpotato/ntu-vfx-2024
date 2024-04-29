import numpy as np
import cv2
import copy

from utils import *

def get_matrix(image):
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
	return Sxx, Sxy, Syy

def Harris_detection(image, k=0.04, threshold=0.001):
	Sxx, Sxy, Syy = get_matrix(image)

	# 4. Define the matrix at each pixel
	# 5. Compute the response of the detect at each pixel
	# -> Instead of doing this pixel wise, we implement in a map wise manner
	detM = Sxx*Syy - Sxy*Sxy
	traceM = Sxx + Syy
	R = detM - k*(traceM**2)

	# 6. Threshold on value of R; compute nonmax suppresion
	sigma = R.max()*threshold
	keypoints = []
	h, w, _ = image.shape
	for i in range(1, h - 1):
		for j in range(1, w - 1):
			pixel = R[i, j]
			neighbor = R[i-1:i+2, j-1:j+2]
			if (pixel > sigma) and ((pixel >= neighbor).all()):
				keypoints.append([i, j])
	
	#plot_keypoints(image, keypoints)

	return keypoints

def MultiScale_Harris_detection(image, scale=2, sigma=1, n=5, threshold=10):
	h, w, _ = image.shape
	# multi-scale
	kernel = (0, 0)
	gaussian_images = []
	gaussian_images.append(image)
	for i in range(1, n):
		gaussian_images.append(cv2.GaussianBlur(image, kernel, sigma**i))

	keypoints = []
	total_R = np.zeros((h, w))
	for i in range(n):
		Sxx, Sxy, Syy = get_matrix(gaussian_images[i])

		detM = Sxx*Syy - Sxy*Sxy
		traceM = Sxx + Syy

		# Harmonic mean
		R = detM / (traceM+1e-6)
		total_R += R

	h, w, _ = image.shape
	total_R /= n

	keypoints = non_maximal_suppression(R)

	#plot_keypoints(image, keypoints)

	return keypoints

def draw_circle(image, center, r):
	h, w = image.shape
	y, x = center
	for i in range(-r, r+1):
		for j in range(-r, r+1):
			if y+i < 0 or y+i >= h or x+j < 0 or x+j >= w:
				continue
			if (i**2 + j**2) <= r**2:
				image[y+i, x+j] = 1

def non_maximal_suppression(R, num=250, r=-1):
	flattened_R = R.flatten()
	sorted_R = np.argsort(flattened_R)[::-1]
	row_indices, col_indices = np.unravel_index(sorted_R, R.shape)
	positions = np.column_stack((row_indices, col_indices))

	keypoints = []
	h, w = R.shape
	if r == -1:
		r = h//40

	filled = np.zeros((h, w))
	while (len(keypoints) < num):
		add = False
		for point in positions:
			y, x = point
			if filled[y, x] == 0:
				keypoints.append(point)
				draw_circle(filled, point, r)
				add = True
				break
		if not add:
			r -= 1
			filled = np.zeros((h, w))
			for point in keypoints:
				draw_circle(filled, point, r)
	return keypoints