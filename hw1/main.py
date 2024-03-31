import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
import os
import natsort
import argparse

import JBF
import HDR
import Reinhard

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
ap.add_argument('-a', '--align', dest='align', type=bool, default=True, help='align or not')
ap.add_argument('-t', '--tone-mapping', dest='toneMapping', type=str, default="Reinhard", help='decide the tone-mapping algorithm to be used')
args = ap.parse_args()

if __name__ == '__main__':
	if args.align:
		_, exposure_time = read_images("image")
		images, _ = read_images("image/align")
	else:
		images, exposure_time = read_images("image")
	images = reshape_images(images, 10)

	# reconstruct HDR image
	B = np.log(exposure_time)
	w = [min(i, 256-i)/128 for i in range(256)]
	g = HDR.recover_response_curve(images, B, w)
	E = HDR.recover_radiance_map(images, B, g, w)

	# tone-mapping to LDR image
	
	# Handcraft Reinhard
	if args.toneMapping == "Reinhard" or args.toneMapping == "all":
		global_output, lm = Reinhard.global_operator(E, 0.8)
		local_output = Reinhard.local_operator(lm, 0.8)
		#cv2.imwrite(f"result/global_tone_mapping_align.png", global_output)
		#cv2.imwrite(f"result/local_tone_mapping_align.png", local_output)
		

	# OpenCV Reinhard
	if args.toneMapping == "OpenCVReinhard" or args.toneMapping == "all":
		E = np.float32(E)
		OpenCVReinhard = cv2.createTonemapReinhard(1.5, 0, 0.5, 0)
		output = OpenCVReinhard.process(E)
		image_show(output)
	
	# Hancraft Bilateral Filter
	if args.toneMapping == "Bilateral" or args.toneMapping == "all":
		bilateral_filter_output = JBF.bilateral_filter(E, 0.4, 5)
		image_show(bilateral_filter_output)
	