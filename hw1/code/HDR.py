import numpy as np
import cv2
import random

def recover_response_curve(images, B, w, l=100):
	# random sample n pixels
	'''
	n = 50
	all_points = [(i, j) for i in range(len(images[0])) for j in range(len(images[0][0]))]
	sampled_points = random.sample(all_points, n)

	Z = np.zeros((3, n, len(images))).astype(int)
	for channel in range(3):
		for i in range(n):
			r, c = sampled_points[i]
			for j in range(len(images)):
				Z[channel,i,j] = images[j,r,c,channel]
	'''
	# down sample
	width, height = 30, 20
	down_sample = []
	for image in images:
		down_sample.append(cv2.resize(image, (width, height)))
	down_sample = np.array(down_sample)
	Z = np.zeros((3, width*height//4, len(images))).astype(int)
	for channel in range(3):
		for i in range(width*height//4):
			r, c = i//(width//2)+height//4, i%(width//2)+width//4
			for j in range(len(images)):
				Z[channel,i,j] = down_sample[j,r,c,channel]


	g = np.zeros((3, 256))
	for i in range(3):
		g[i] = gsolve(Z[i], B, l, w)
	return g


def gsolve(Z, B, l, w):
	'''
	Z[i, j]: value of ith pixel in image j
	B      : exposure time, ln delta t
	l      : lambda, parameter to regularize, make g smoother
	w      : weight function
	'''
	n, p = Z.shape
	c = 256
	A = np.zeros((n*p+(c - 2)+1, c+n))
	b = np.zeros((A.shape[0], 1))

	# fill A
	k = 0  # row in A
	for i in range(n):
		for j in range(p):
			wij = w[Z[i, j]]
			A[k, Z[i, j]] = wij
			A[k, c+i] = -wij
			b[k, 0] = wij*B[j]
			k += 1

	A[k, 127] = 1
	k += 1

	for i in range(c - 2):
		A[k, i] = l*w[i+1]
		A[k, i+1] = -2*l*w[i+1]
		A[k, i+2] = l*w[i+1]
		k += 1

	x = np.linalg.lstsq(A, b, rcond=None)[0]  # return the least-squaure solution

	g = x[:c].reshape(-1)
	lE = x[c:].reshape(-1)

	return g

def recover_radiance_map(images, B, g, w, ghost_removal=False):
	p = len(images)
	height, width, _ = images[0].shape
	E = np.zeros((height, width, 3))
	# for each pixel in an image
	for channel in range(3):
		for r in range(height):
			for c in range(width):
				lE = 0
				w_sum = 0
				if ghost_removal:
					lE_avg = 0
					for j in range(p):
						zij = images[j,r,c,channel]
						lE_avg += g[channel,zij]-B[j]
					lE_avg /= p
					for j in range(p):
						zij = images[j,r,c,channel]
						lE += w[zij]/(abs((g[channel,zij]-B[j])-lE_avg)**2)*(g[channel,zij]-B[j])
						w_sum += w[zij]/(abs((g[channel,zij]-B[j])-lE_avg)**2)
				else:
					for j in range(p):
						zij = images[j,r,c,channel]
						lE += w[zij]*(g[channel,zij]-B[j])
						w_sum += w[zij]
				if w_sum != 0:
					lE /= w_sum
				E[r,c,channel] = lE
	E = np.exp(E)
	return E