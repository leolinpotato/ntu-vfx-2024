import numpy as np
import cv2
from scipy.spatial.distance import cdist

from utils import *

def feature_matching(test_descriptors, train_descriptors):
	test_desc = [test_descriptor['descriptor'] for test_descriptor in test_descriptors]
	train_desc = [train_descriptor['descriptor'] for train_descriptor in train_descriptors]
	distance = cdist(test_desc, train_desc)
	distance_sorted = np.argsort(distance, axis=1)
	match = []
	match_id = []
	for i in range(len(test_descriptors)):
		first = distance[i, distance_sorted[i, 0]]
		second = distance[i, distance_sorted[i, 1]]
		if (first/second < 0.8):
			match.append([test_descriptors[i], train_descriptors[distance_sorted[i, 0]]])
			match_id.append((i, distance_sorted[i, 0]))
	return match, match_id