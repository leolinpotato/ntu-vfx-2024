import numpy as np
import cv2
import os
import natsort
import argparse
import time

from utils import *
from feature_detection import *
from feature_descriptor import *
from feature_matching import *
#from image_matching import *
from cylinder_cordinate import *
from blending import *

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dir', dest='dir', type=str, default="../data/CKS_memorial_hall", help='the input image directory')
args = ap.parse_args()

def keypoints_descriptors(image, type='Train'):
	t0 = time.time()
	keypoints = MultiScale_Harris_detection(image)
	print("detection", time.time() - t0)
	t0 = time.time()
	descriptors = SIFT_descriptor(image, keypoints)
	print("descriptor", time.time() - t0)
	#print(f"{type} keypoints: ", len(keypoints))
	#print(f"{type} descriptor:", len(descriptors))
	return keypoints, descriptors

# def read_focal(n, dir="../data/test/focal_length.txt"):
# 	fl_file = open(dir)
# 	lines = fl_file.readlines()
# 	focal_lengths = [0 for i in range(n)]
# 	for line in lines:
# 		tmp = line.replace('\n', '')
# 		tmp = tmp.split(' ')
# 		if (len(tmp) < 2):
# 			continue
# 		#print(tmp)
# 		i = int(tmp[0])
# 		fl = float(tmp[1])
# 		focal_lengths[i] = fl
# 	return focal_lengths

def read_focal(n, path="../data/parrington/pano.txt"):
	with open(path, 'r') as file:
		content = file.read()
		lines = content.splitlines()
	# idx: the id of current line
	idx = 0
	file_focal = []
	for i in range(n):
		file_name = lines[idx].split('\\')[-1]
		idx += 11
		focal_length = float(lines[idx])
		idx += 2
		file_focal.append((file_name, focal_length))
	file_focal = sorted(file_focal, key=lambda x: x[0])
	focal_lengths = [focal_length[-1] for focal_length in file_focal]
	return focal_lengths



if __name__ == '__main__':
	##show_ransac(args.dir)
	
	images = read_images(args.dir)


	#image_to_cylinder(images[0], 704.867)
	#read focal length
	#focal_lengths = read_focal(len(images))
	#imgs_cylinder = images_to_cylinder(images, focal_lengths)
	#show_ransac(imgs_cylinder[0], imgs_cylinder[1])


	# images_reshape = [reshape_image(img, 4) for img in images]
	# focal_lengths = read_focal(18)
	# cylinder_imgs = images_to_cylinder(images_reshape, focal_lengths)



	#images_reshape.reverse()
	'''
	for i in range(len(images_reshape)-1) :
		show_ransac(images_reshape[i], images_reshape[i + 1])
	'''
	#blending(images_reshape)
	'''
	show_ransac(images_reshape[0], images_reshape[1])
	images_matching = image_matching(images_reshape, cheat = True)
	for img_match in images_matching:
		print("len:", len(img_match))
	print("total_img:", len(images_matching))
	'''
	
	'''
	for img_mt in images_matching:
		print(img_mt)
		print(len(img_mt))
		'''
	train_image = reshape_image(images[0], 10)
	train_keypoints, train_descriptors = keypoints_descriptors(train_image, 'Train')
'''
	test_image = reshape_image(images[1], 3)
	test_keypoints, test_descriptors = keypoints_descriptors(test_image, 'Test')

	
	#print(image_matching(train_image, test_image))
	
	ransac_match = ransac(test_image, train_image)
	ransac_matches, ransac_matches_id = zip(*ransac_match)
	print("num_imnlier:", len(ransac_match))
	
	matches, matches_id = feature_matching(test_descriptors, train_descriptors)
	matches_id = np.array(matches_id)
	#matches = np.array(matches)
	print("Match:", len(matches))
	#print(len(train_keypoints), len(test_keypoints), max(matches_id[:, 0]))

	# convert to cv2 data type
	train_keypoints_position = [cv2.KeyPoint(x=desc['point'][0], y=desc['point'][1], size=10) for desc in train_descriptors]	
	test_keypoints_position = [cv2.KeyPoint(x=desc['point'][0], y=desc['point'][1], size=10) for desc in test_descriptors]
	ransac_matches_id = [cv2.DMatch(_queryIdx=match[0], _trainIdx=match[1], _distance=0) for match in ransac_matches_id]	
	#print(matches_id[0].queryIdx, matches_id[0].trainIdx)
	matched_img = cv2.drawMatches(test_image, test_keypoints_position, train_image, train_keypoints_position, ransac_matches_id, None)
	image_show(matched_img)
	'''