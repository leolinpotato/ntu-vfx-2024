import numpy as np
import cv2

def global_operator(lw, a=0.8, l_white=2.5):
	delta = 1e-10
	# B, G, R = 0.114, 0.587, 0.299
	B, G, R = 0.06, 0.67, 0.27
	lw_bar = np.exp(np.mean(np.log(delta+lw[:,:,0])*B + np.log(delta+lw[:,:,1])*G + np.log(delta+lw[:,:,2])*R))
	lm = lw*a/lw_bar
	# print(np.array(lm).max())
	ld = lm*(1+lm/l_white**2)/(1+lm)
	output = np.clip(ld*255, 0, 255).astype(np.uint8)
	return output, lm

def local_operator(lm, a=0.8):
	# the parameter setting is based on the paper "Erik Reinhard, Michael Stark, Peter Shirley, Jim Ferwerda, Photographics Tone Reproduction for Digital Images, SIGGRAPH 2002."
	phi = 10
	threshold = 0.05
	l_blur = []
	scale = []
	pixel = 1.0
	for i in range(8):
		l_blur.append(cv2.GaussianBlur(lm, (int(2*np.rint(pixel)+1), int(2*np.rint(pixel)+1)), 0, 0))
		scale.append(int(2*np.rint(pixel)+1))
		pixel *= 1.6
	# find Smax
	Smax = 0
	for s in range(len(l_blur)-1):
		V = (l_blur[s]-l_blur[s+1])/((2**phi)*a/(scale[s]**2)+l_blur[s])
		if np.all(abs(V) < threshold):
			Smax = s
	ld = lm/(1+l_blur[Smax])
	output = np.clip(ld*255, 0, 255).astype(np.uint8)
	return output