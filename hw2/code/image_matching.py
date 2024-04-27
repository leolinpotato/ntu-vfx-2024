import numpy as np
import cv2
import os
import natsort
import argparse
import math

from utils import *
from feature_detection import *
from feature_descriptor import *
from feature_matching import *
from main import keypoints_descriptors




def alignment_2d(matches):
    '''
    align two images by its matching using matrix
    a b
    c d
    '''
    N = len(matches)
    A = np.zeros((N * 2, 4))
    i = 0
    b = np.zeros(N * 2)
    for match in matches:
        A[i][0] = match[0]['point'][0]
        A[i][1] = match[0]['point'][1]
        i += 1
        A[i][2] = match[0]['point'][0]
        A[i][3] = match[0]['point'][1]
        i += 1
    i = 0
    for match in matches:
        b[i] = match[1]['point'][0]
        i += 1
        b[i] = match[1]['point'][1]
        i += 1
    A_t = np.transpose(A)
    A_t_A = np.matmul(A_t, A)
    A_t_b = np.matmul(A_t, b)
    try:
        m = np.linalg.solve(A_t_A, A_t_b)
    except np.linalg.LinAlgError:
        m = np.linalg.pinv(A_t_A) @ A_T_b
    ret_m = [[m[0], m[1]], [m[2], m[3]]]
    #print(ret_m)
    return ret_m
def alignment_affine(matches):
    '''
    align two images by its matching using matrix
    a b c
    d e f
    0 0 1
    '''
    if len(matches) == 3:
        N = len(matches)
        A = np.zeros((N * 2, 6))
        i = 0
        b = np.zeros(N * 2)
        for match in matches:
            A[i][0] = match[0]['point'][0]
            A[i][1] = match[0]['point'][1]
            A[i][2] = 1
            i += 1
            A[i][3] = match[0]['point'][0]
            A[i][4] = match[0]['point'][1]
            A[i][5] = 1
            i += 1
        i = 0
        for match in matches:
            b[i] = match[1]['point'][0]
            i += 1
            b[i] = match[1]['point'][1]
            i += 1
        try:
            m = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            m = np.linalg.pinv(A) @ b
        ret_m = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [0, 0, 1]]
        #print(ret_m)
        return ret_m
    N = len(matches)
    A = np.zeros((N * 2, 6))
    i = 0
    b = np.zeros(N * 2)
    for match in matches:
        A[i][0] = match[0]['point'][0]
        A[i][1] = match[0]['point'][1]
        A[i][2] = 1
        i += 1
        A[i][3] = match[0]['point'][0]
        A[i][4] = match[0]['point'][1]
        A[i][5] = 1
        i += 1
    i = 0
    for match in matches:
        b[i] = match[1]['point'][0]
        i += 1
        b[i] = match[1]['point'][1]
        i += 1
    A_t = np.transpose(A)
    A_t_A = np.matmul(A_t, A)
    A_t_b = np.matmul(A_t, b)
    try:
        m = np.linalg.solve(A_t_A, A_t_b)
    except np.linalg.LinAlgError:
        m = np.linalg.pinv(A_t_A) @ A_t_b
    ret_m = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [0, 0, 1]]
    #print(ret_m)
    return ret_m
def alignment_affine_no_turn(matches):
    '''
    align two images by its matching using matrix
    a 0 b
    0 c d
    0 0 1
    '''
    N = len(matches)
    A = np.zeros((N * 2, 4))
    i = 0
    b = np.zeros(N * 2)
    for match in matches:
        A[i][0] = match[0]['point'][0]
        #A[i][1] = match[0]['point'][1]
        A[i][1] = 1
        i += 1
        A[i][2] = match[0]['point'][1]
        #A[i][4] = match[0]['point'][1]
        A[i][3] = 1
        i += 1
    i = 0
    for match in matches:
        b[i] = match[1]['point'][0]
        i += 1
        b[i] = match[1]['point'][1]
        i += 1
    A_t = np.transpose(A)
    A_t_A = np.matmul(A_t, A)
    A_t_b = np.matmul(A_t, b)
    try:
        m = np.linalg.solve(A_t_A, A_t_b)
    except np.linalg.LinAlgError:
        m = np.linalg.pinv(A_t_A) @ A_t_b
    ret_m = [[m[0], 0, m[1]], [0, m[2], m[3]], [0, 0, 1]]
    #print(ret_m)
    return ret_m


    
def ransac(img1, img2):
    #use ransac to find best inlier and shift for two images
    #img1_keypoints = Harris_detection(img1)
    #img2_keypoints = Harris_detection(img2)
    img1_keypoints, img1_descriptors = keypoints_descriptors(img1, 'img1')
    img2_keypoints, img2_descriptors = keypoints_descriptors(img2, 'img2')
    #img1_descriptors = SIFT_descriptor(img1, img1_keypoints)
    #img2_descriptors = SIFT_descriptor(img2, img2_keypoints)
    matches, matches_id = feature_matching(img1_descriptors, img2_descriptors)
    #print("mtsize:", len(matches))
    mt_copy = matches
    np.random.shuffle(mt_copy)
    k = 1000 #time of iteration for ransac
    n = 3 #num of point select a time
    max_cnt = 0
    best_inliers = []
    best_inliers_match = []
    best_shift = []
    tolerence = img1.shape[1] / 15 # tolerence
    print("tol:", tolerence)
    for i in range(k):
        np.random.shuffle(mt_copy)
        mt_arg = mt_copy[0:n]
        average_length = 0
        
        for mt in mt_arg:
            x1, y1 = mt[0]['point']
            x2, y2 = mt[1]['point']
            #x_max = img1.shpae[2]
            average_length += x1 - x2
        average_length /= n
        tol1 = 1
        out = False
        for mt in mt_arg:
            x1, y1 = mt[0]['point']
            x2, y2 = mt[1]['point']
            if abs(average_length - (x1-x2)) > tol1:
                out = True
                #print("skip")
                break
            #print("pass", x1, y1, x2, y2)
        #print("------")
        if (out):
            continue
            
        shift = alignment_affine(mt_arg)
        shift = np.array(shift)
        inlier_cnt = 0
        inliers = []
        inliers_match = []
        for match, match_id in zip(matches, matches_id):
            x1, y1 = match[0]['point']
            x2, y2 = match[1]['point']
            cor = np.array([x1, y1, 1])
            shift_cor = np.matmul(shift, cor)
            x1_ = shift_cor[0]
            y1_ = shift_cor[1]
            dif = math.sqrt((x1_ - x2)**2 + (y1_ - y2)**2)
            if dif < tolerence:
                #print("in inl", x1, y1, x2, y2)
                inlier_cnt += 1
                inliers.append((match, match_id))
                inliers_match.append(match)
        if inlier_cnt > max_cnt:
            max_cnt = inlier_cnt
            #print("updated-------------------------------------")
            best_inliers_match = inliers_match
            best_inliers = inliers
            best_shift = shift
    #print(best_shift)
    
    best_shift = alignment_affine_no_turn(best_inliers_match)
    return best_inliers, best_shift
def len_1d(pair):
    a, b, c = pair
    return len(a)

def x_in_y(x, y):
    if x[0] > 0 and x[1] > 0 and x[0] < y[0] and x[1] < y[1]:
        return True
    return False

def image_matching(images, cheat=False):
    # match images 
    # reture list of list of tuple (matches, matches_id, j, best_inlier, best_shift)
    i = 0
    m = 6
    shape_x, shape_y, _ = images[0].shape
    imgs_matched = []
    images_ = images
    for img1 in images:
        this_img_mt = []
        j = 0
        n = len(images)
        img1_keypoints, img1_descriptors = keypoints_descriptors(img1, 'img1')
        if (cheat):
            for j in range(len(images)):
                if (i != j and ((j + 1) % n == i or (i + 1) % n == j)) :
                    img2 = images[j]
                    img2_keypoints, img2_descriptors = keypoints_descriptors(img2, 'img2')
                    mt, mt_id = feature_matching(img1_descriptors, img2_descriptors)
                    #print("match", i, j, len(mt))
                    this_img_mt.append((mt, mt_id, j))
                    j += 1
            candidate_best_inlier = []
            for candidate in this_img_mt:
                matches, matches_id, j = candidate
                best_inlier, best_shift = ransac(images[j], img1)
                if (len(best_shift) == 0):
                    print("skipped", i, j)
                    continue
                nf = 0
                ni = len(best_inlier)
                for mt in matches:
                    point_cand = list(mt[0]['point'])
                    point_cand.append(1)
                    point_cand = np.array(point_cand)
                    project_point = np.matmul(best_shift, point_cand)
                    if x_in_y(project_point, (shape_x, shape_y)) :
                        nf += 1
                print("accept", i, j)
                candidate_best_inlier.append((matches, matches_id, j, best_inlier, best_shift))
                
        else :
            for j in range(len(images)):
                if (i != j) :
                    img2 = images[j]
                    img2_keypoints, img2_descriptors = keypoints_descriptors(img2, 'img2')
                    mt, mt_id = feature_matching(img1_descriptors, img2_descriptors)
                    this_img_mt.append((mt, mt_id, j))
                    j += 1
            this_img_mt.sort(key = len_1d, reverse = True)
            candidate_best_inlier = []
            for candidate in this_img_mt[0:6]:
                matches, matches_id, j = candidate
                best_inlier, best_shift = ransac(images[j], img1)
                if (len(best_shift) == 0):
                    print("skipped", i, j)
                    continue
                nf = 0
                ni = len(best_inlier)
                for mt in matches:
                    point_cand = list(mt[0]['point'])
                    point_cand.append(1)
                    point_cand = np.array(point_cand)
                    project_point = np.matmul(best_shift, point_cand)
                    if x_in_y(project_point, (shape_x, shape_y)) :
                        nf += 1
                if (ni > 5.9 + 0.22 * nf):
                    print("accept", i, j)
                    candidate_best_inlier.append((matches, matches_id, j, best_inlier, best_shift))
                else:
                    print("reject", i, j)
        
        imgs_matched.append(candidate_best_inlier)
        i += 1
    return imgs_matched
    

def show_ransac(train_image, test_image):
    #images = read_images(dir)

    #train_image = reshape_image(images[0], 3)
    train_keypoints, train_descriptors = keypoints_descriptors(train_image, 'Train')

    #test_image = reshape_image(images[1], 3)
    test_keypoints, test_descriptors = keypoints_descriptors(test_image, 'Test')

    mt, mt_id = feature_matching(test_descriptors, train_descriptors)
    print("num_mt:", len(mt))
    #print(image_matching(train_image, test_image))

    ransac_match, _ = ransac(test_image, train_image)
    ransac_matches, ransac_matches_id = zip(*ransac_match)
    print("num_imnlier:", len(ransac_match))
    for mt in ransac_matches:
        x1, y1 = mt[0]['point']
        x2, y2 = mt[1]['point']
        print("ransac_match", x1, y1, x2, y2)
    '''
    matches, matches_id = feature_matching(test_descriptors, train_descriptors)
    matches_id = np.array(matches_id)
    #matches = np.array(matches)
    print("Match:", len(matches))
    #print(len(train_keypoints), len(test_keypoints), max(matches_id[:, 0]))
    '''

    # convert to cv2 data type
    train_keypoints_position = [cv2.KeyPoint(x=desc['point'][0], y=desc['point'][1], size=10) for desc in train_descriptors]	
    test_keypoints_position = [cv2.KeyPoint(x=desc['point'][0], y=desc['point'][1], size=10) for desc in test_descriptors]
    ransac_matches_id = [cv2.DMatch(_queryIdx=match[0], _trainIdx=match[1], _distance=0) for match in ransac_matches_id]	
    #print(matches_id[0].queryIdx, matches_id[0].trainIdx)
    matched_img = cv2.drawMatches(test_image, test_keypoints_position, train_image, train_keypoints_position, ransac_matches_id, None)
    print(_)
    image_show(matched_img)

    
    



