import numpy as np
import cv2
import time

from utils import *

def get_magnitude_theta(image):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	kernel = (5, 5)
	GI = cv2.GaussianBlur(gray_image, kernel, 0)
	Iy, Ix = np.gradient(GI)
	magnitude = np.sqrt(Ix**2 + Iy**2)
	theta = np.arctan2(Ix, Iy)*180/np.pi

	# theta is between -180 to 180, so needs to be normalized to 0 to 360
	theta[theta<0] = theta[theta<0]+360

	return magnitude, theta

def orientation_assignment(image, keypoints):
	magnitude, theta = get_magnitude_theta(image)

	new_keypoints = []
	orientations = []
	h, w, _ = image.shape
	patch_size = 3
	sigma = 1.5
	Gs = np.array([[np.exp(-((i - patch_size)**2 + (j - patch_size)**2) / (2 * sigma**2)) for i in range(patch_size*2+1)] for j in range(patch_size*2+1)])

	bins = 36
	bin_size = 360 / bins
 
	for point in keypoints:
		y, x = point
		if (x-patch_size < 0) or (x+patch_size >= w) or (y-patch_size < 0) or (y+patch_size >= h):
		    continue
		histogram = np.zeros(bins)
		for i in range(y-patch_size, y+patch_size+1):
			for j in range(x-patch_size, x+patch_size+1):
				b = int(np.round(theta[i, j]/bin_size)) % bins
				histogram[b] += magnitude[i, j]*Gs[i-(y-patch_size), j-(x-patch_size)]
		peak = np.max(histogram)
		for i, his in enumerate(histogram):
			if his >= peak*0.8:
				new_keypoints.append(point)
				orientations.append(i*bin_size)

	return new_keypoints, orientations

# descriptors: a dict with 2-d point position and a 128 dimensional descriptor
def SIFT_descriptor(image, keypoints):
	h, w, _ = image.shape
	keypoints, orientations = orientation_assignment(image, keypoints)

	descriptors = []
	bins = 8
	bin_size = 360 / bins

	for idx in range(len(keypoints)):
		y, x = keypoints[idx]
		x, y = int(x), int(y)
		if (x-8 < 0) or (x+8 > w) or (y-8 < 0) or (y+8 > h):
			continue
		descriptor = []
		image_rotation = rotate_image(image, -orientations[idx], (x, y))
		magnitude, theta = get_magnitude_theta(image_rotation)
		for i in range(y-8, y+8, 4):
			for j in range(x-8, x+8, 4):
				histogram = np.zeros(bins)
				for dy in range(4):
					for dx in range(4):
						b = int(np.round(theta[i+dy, j+dx]/bin_size)) % bins
						histogram[b] += magnitude[i+dy, j+dx]
				descriptor.extend(histogram)
		descriptor = normalize(descriptor)
		descriptor[descriptor>0.2] = 0.2
		descriptor = normalize(descriptor)
		descriptors.append({'point': (x, y), 'descriptor': descriptor})
	return descriptors
