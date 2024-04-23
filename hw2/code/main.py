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

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dir', dest='dir', type=str, default="../data/test", help='the input image directory')
args = ap.parse_args()

if __name__ == '__main__':
	images = read_images(args.dir)

	train_image = reshape_image(images[0], 10)
	train_keypoints = MultiScale_Harris_detection(train_image)
	train_descriptors = SIFT_descriptor(train_image, train_keypoints)
	print(f"Train keypoints: ", len(train_keypoints))
	print(f"Train descriptor:", len(train_descriptors))

	test_image = reshape_image(images[1], 10)
	test_keypoints = MultiScale_Harris_detection(test_image)
	test_descriptors = SIFT_descriptor(test_image, test_keypoints)
	print(f"Test keypoints: ", len(test_keypoints))
	print(f"Test descriptor:", len(test_descriptors))	

	matches, matches_id = feature_matching(test_descriptors, train_descriptors)
	print("Match:", len(matches))

	# convert to cv2 data type
	train_keypoints_position = [cv2.KeyPoint(x=desc['point'][0], y=desc['point'][1], size=10) for desc in train_descriptors]	
	test_keypoints_position = [cv2.KeyPoint(x=desc['point'][0], y=desc['point'][1], size=10) for desc in test_descriptors]
	matches_id = [cv2.DMatch(_queryIdx=match[0], _trainIdx=match[1], _distance=0) for match in matches_id]	
	matched_img = cv2.drawMatches(test_image, test_keypoints_position, train_image, train_keypoints_position, matches_id, None)
	image_show(matched_img)