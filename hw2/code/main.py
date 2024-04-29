import numpy as np
import cv2
import os
import natsort
import argparse

from utils import *
from feature_detection import *
from feature_descriptor import *
from feature_matching import *
#from image_matching import *
from cylinder_cordinate import *
from blending import *

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dir', dest='dir', type=str, default="../data/test", help='the input image directory')
args = ap.parse_args()

def keypoints_descriptors(image, type='Train'):
	keypoints = MultiScale_Harris_detection(image, threshold = 5)
	descriptors = SIFT_descriptor(image, keypoints)
	#print(f"{type} keypoints: ", len(keypoints))
	#print(f"{type} descriptor:", len(descriptors))
	return keypoints, descriptors

def read_focal(n, ratio, dir="../data/test/focal_length.txt"):
	fl_file = open(dir)
	lines = fl_file.readlines()
	focal_lengths = [0 for i in range(n)]
	for line in lines:
		tmp = line.replace('\n', '')
		tmp = tmp.split(' ')
		if (len(tmp) < 2):
			continue
		#print(tmp)
		i = int(tmp[0])
		fl = float(tmp[1])
		focal_lengths[i] = 4700 / ratio
	return focal_lengths

if __name__ == '__main__':
	##show_ransac(args.dir)
	
	images = read_images(args.dir)
	#read focal length
	ratio = 10
	focal_lengths = read_focal(len(images), ratio)
	images_reshape = [reshape_image(img, ratio) for img in images]
	i = 0
	'''
	for img in images_reshape:
		cv2.imwrite('compress/img_'+ str(i) + '.jpg', img)
		i += 1
	'''
	#image_show(image_to_cylinder(images_reshape[0], focal_lengths[0]))
	#images_reshape = [reshape_image(img, ratio) for img in images]
	images_cylinder = images_to_cylinder(images_reshape, focal_lengths)
	
	
	
	'''
	for img in images_cylinder:
		image_show(img)
		'''
	#image_show(images_cylinder[0])
	
	
	blending(images_cylinder)
	
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
	'''
	train_image = reshape_image(images[1], 3)
	train_keypoints, train_descriptors = keypoints_descriptors(train_image, 'Train')
	test_image = reshape_image(images[2], 3)
	test_keypoints, test_descriptors = keypoints_descriptors(test_image, 'Test')

	
	#print(image_matching(train_image, test_image))
	
	matches, matches_id = feature_matching(test_descriptors, train_descriptors)
	matches_id = np.array(matches_id)
	#matches = np.array(matches)
	print("Match:", len(matches))
	#print(len(train_keypoints), len(test_keypoints), max(matches_id[:, 0]))

	# convert to cv2 data type
	train_keypoints_position = [cv2.KeyPoint(x=desc['point'][0], y=desc['point'][1], size=10) for desc in train_descriptors]	
	test_keypoints_position = [cv2.KeyPoint(x=desc['point'][0], y=desc['point'][1], size=10) for desc in test_descriptors]
	matches_id = [cv2.DMatch(_queryIdx=match[0], _trainIdx=match[1], _distance=0) for match in matches_id]	
	#print(matches_id[0].queryIdx, matches_id[0].trainIdx)
	matched_img = cv2.drawMatches(test_image, test_keypoints_position, train_image, train_keypoints_position, matches_id, None)
	image_show(matched_img)
	#show_ransac(imgs_cylinder[0], imgs_cylinder[1])
	#images_reshape = [reshape_image(img, 30) for img in imgs_cylinder]
	#images_reshape.reverse()
	'''