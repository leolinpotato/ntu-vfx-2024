import numpy as np
import cv2
import os
import natsort

def image_show(image):
	cv2.imshow("image", image)
	cv2.waitKey(0)

def reshape_images(images, ratio=1):
	for i in range(len(images)):
		h, w, _ = images[i].shape
		images[i] = cv2.resize(images[i], (w//ratio, h//ratio))
	images = np.array(images)
	return images

def reshape_image(image, ratio=1):
	h, w, _ = image.shape
	image = cv2.resize(image, (w//ratio, h//ratio))
	return image

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

	return images

def rotate_image(image, orientation, center=None):
	h, w, _ = image.shape
	if center == None:
		center = (w / 2, h / 2)
	M = cv2.getRotationMatrix2D(center, orientation, 1)
	image_rotation = cv2.warpAffine(image, M, (w, h))
	return image_rotation

def normalize(n):
	norm = np.linalg.norm(n)
	if (norm == 0):
		return n
	return n / norm