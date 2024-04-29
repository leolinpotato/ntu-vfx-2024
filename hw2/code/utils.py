import numpy as np
import cv2
import os
import natsort
import copy


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

def plot_keypoints(image, keypoints):
	highlight_image = copy.deepcopy(image)
	h, w, _ = image.shape
	sz = max(1, w // 500)
	for point in keypoints:
		x, y = point
		highlight_image[x-sz:x+sz+1, y-sz:y+sz+1] = [0, 0, 255]
	image_show(highlight_image)
	cv2.imwrite("../result/duck.png", highlight_image)

def top_n_values_with_indices(arr, n):
    # Flatten the array
    flattened_arr = arr.flatten()
    
    # Sort the flattened array in descending order
    sorted_indices = np.argsort(flattened_arr)[::-1]
    
    # Get the top n indices
    top_n_indices = sorted_indices[:n]
    
    # Get the top n values
    top_n_values = flattened_arr[top_n_indices]
    
    # Get the row and column indices corresponding to the flattened indices
    row_indices, col_indices = np.unravel_index(top_n_indices, arr.shape)
    
    # Combine row and column indices
    positions = np.column_stack((row_indices, col_indices))
    return positions