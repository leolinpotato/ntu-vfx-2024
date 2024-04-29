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
from cylinder_cordinate import *
from blending import *

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dir', dest='dir', type=str, default="../data/CKS_memorial_hall", help='the input image directory')
args = ap.parse_args()

def keypoints_descriptors(image):
	keypoints = MultiScale_Harris_detection(image)
	descriptors = SIFT_descriptor(image, keypoints)
	return keypoints, descriptors

def read_focal(n, ratio, dir="../focal_length.txt"):
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
	images = read_images(args.dir)

	#read focal length
	ratio = 10
	focal_lengths = read_focal(len(images), ratio)
	images_reshape = [reshape_image(img, ratio) for img in images]

	images_cylinder = images_to_cylinder(images_reshape, focal_lengths)	
	blending(images_cylinder)

	'''
	##### Draw match #####

	train_image = reshape_image(images[1], 3)
	train_keypoints, train_descriptors = keypoints_descriptors(train_image, 'Train')
	test_image = reshape_image(images[2], 3)
	test_keypoints, test_descriptors = keypoints_descriptors(test_image, 'Test')

	matches, matches_id = feature_matching(test_descriptors, train_descriptors)

	# convert to cv2 data type
	train_keypoints_position = [cv2.KeyPoint(x=desc['point'][0], y=desc['point'][1], size=10) for desc in train_descriptors]	
	test_keypoints_position = [cv2.KeyPoint(x=desc['point'][0], y=desc['point'][1], size=10) for desc in test_descriptors]
	matches_id = [cv2.DMatch(_queryIdx=match[0], _trainIdx=match[1], _distance=0) for match in matches_id]	
	matched_img = cv2.drawMatches(test_image, test_keypoints_position, train_image, train_keypoints_position, matches_id, None)
	image_show(matched_img)
	'''