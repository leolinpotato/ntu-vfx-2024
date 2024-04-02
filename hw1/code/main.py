import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
import os
import natsort
import argparse
import matplotlib.pyplot as plt

import JBF
import HDR
import Reinhard
import HDR_MitsuagaAndNayar as HDR_MAN
import alignment

def plot_HDR(E):
	B, G, R = 0.06, 0.67, 0.27
	intensity = (E[:,:,0]*B + E[:,:,1]*G + E[:,:,2]*R).astype(np.float32)
	plt.figure(figsize=(8, 4))
	plt.imshow(intensity, cmap='jet')
	plt.xticks([])
	plt.yticks([])
	plt.colorbar()
	plt.tight_layout()
	plt.savefig('../data/hdr.png')
	plt.close()

def plot_response_curve(g, l):
	plt.figure(figsize=(6, 4))
	plt.title(f'Response Curve (lambda = {l})')
	plt.plot(g[0], range(256), color='b', marker='.', markersize=1)
	plt.plot(g[1], range(256), color='g', marker='.', markersize=1)
	plt.plot(g[2], range(256), color='r', marker='.', markersize=1)
	plt.xlabel('log exposure X')
	plt.ylabel('pixel value Z')
	plt.tight_layout()
	plt.savefig(f'../data/response_curve_{l}.png')
	plt.close()

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
		exposure_time.append(np.float32(get_exif_data(image_path)))

	exposure_time = np.array(exposure_time)
	return images, exposure_time

def get_exif_data(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Get the EXIF data
        exif_data = img._getexif()
        if exif_data is not None:
            # Iterate over all EXIF tags
            for tag, value in exif_data.items():
                # Decode the tag
                tag_name = TAGS.get(tag, tag)
                # Check if the tag is 'ExposureTime'
                if tag_name == 'ExposureTime':
                    # Return the exposure time
                    return value

def reshape_images(images, ratio=1):
	for i in range(len(images)):
		h, w, _ = images[i].shape
		images[i] = cv2.resize(images[i], (w//ratio, h//ratio))
	images = np.array(images)
	return images

def image_show(image):
	cv2.imshow("image", image)
	cv2.waitKey(0)


ap = argparse.ArgumentParser()
ap.add_argument('-a', '--align', dest='align', action="store_true", help='align or not')
ap.add_argument('-d', '--hdr', dest='HDR', type=str, default="PaulDebevec", help='decide the HDR algorithm to be used')
ap.add_argument('-t', '--tone-mapping', dest='toneMapping', type=str, default="all", help='decide the tone-mapping algorithm to be used')
ap.add_argument('-g', '--ghost-removal', dest='ghostRemoval', action="store_true", help='ghost removal or not')
ap.add_argument('-p', '--plot', dest='plot', action="store_true", help='plot HDR and response_curve (only for -d PaulDebevec option) or not')
args = ap.parse_args()

if __name__ == '__main__':
	if args.align:
		images, exposure_time = read_images("../data/image")
		images = alignment.alignment_main(images, "../data/image/align")
		#images, _ = read_images("../data/image/align")
	else:
		images, exposure_time = read_images("../data/image")
	images = reshape_images(images, 10)

	# reconstruct HDR image
	if args.HDR == "PaulDebevec":

		B = np.log(exposure_time)
		w = [min(i, 256-i)/128 for i in range(256)]
		l = 100
		g = HDR.recover_response_curve(images, B, w, l)
		E = HDR.recover_radiance_map(images, B, g, w, args.ghostRemoval)
		# cv2.imwrite("../data/HDR_images/hdr_PaulDebevec.hdr", E)
		if args.plot:
			path = "../data/HDR_images/hdr_PaulDebevec.hdr"
			plot_HDR(E)
			plot_response_curve(g, l)
	elif args.HDR == "MitsuagaNayar":
		N = 600 // 4 # number of point selected
		M = 5 # degree of polynomial, not exceed 10
		w_x = 220 # the min of weight
		E = HDR_MAN.MitsuagaAndNayar_HDR(images, N, M, w_x)
		# cv2.imwrite("../data/HDR_images/hdr_MitsuagaNayar.hdr", E)
		if args.plot:
			plot_HDR(E)

		

	# tone-mapping to LDR image
	
	# Handcraft Reinhard
	if args.toneMapping == "Reinhard" or args.toneMapping == "all":
		global_output, lm = Reinhard.global_operator(E, 0.8)
		local_output = Reinhard.local_operator(lm, 0.8)
		image_show(global_output)
		image_show(local_output)
		# cv2.imwrite(f"../data/tone_mapped_images/ReinhardGlobal.png", global_output)
		# cv2.imwrite(f"../data/tone_mapped_images/ReinhardLocal.png", local_output)
		

	# OpenCV Reinhard
	if args.toneMapping == "OpenCVReinhard" or args.toneMapping == "all":
		E = np.float32(E)
		OpenCVReinhard = cv2.createTonemapReinhard(1.5, 0, 0.5, 0)
		output = OpenCVReinhard.process(E)
		image_show(output)
		# cv2.imwrite(f"../data/tone_mapped_images/OpenCVReinhard.png", output*255)
	
	# OpenCV Drago
	if args.toneMapping == "OpenCVDrago" or args.toneMapping == "all":
		E = np.float32(E)
		OpenCVDrago = cv2.createTonemapDrago(1.5, 1, 0.85)
		output = OpenCVDrago.process(E)
		image_show(output)
		# cv2.imwrite(f"../data/tone_mapped_images/OpenCVDrago.png", output*255)

	# OpenCV Mantiuk
	if args.toneMapping == "OpenCVMantiuk" or args.toneMapping == "all":
		E = np.float32(E)
		OpenCVMantiuk = cv2.createTonemapMantiuk(1.5, 0.7, 1)
		output = OpenCVMantiuk.process(E)
		image_show(output)
		# cv2.imwrite(f"../data/tone_mapped_images/OpenCVMantiuk.png", output*255)

	# Hancraft Bilateral Filter
	if args.toneMapping == "Bilateral" or args.toneMapping == "all":
		bilateral_filter_output = JBF.bilateral_filter(E, 0.4, 5)
		image_show(bilateral_filter_output)
		# cv2.imwrite(f"../data/tone_mapped_images/BilateralFilter.png", bilateral_filter_output)
	