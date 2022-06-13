import cv2
import numpy as np


def extract_blue(imag):
	mser_blue = cv2.MSER_create(8, 400, 4000)

	img = imag.copy()
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	# equalize the histogram of the Y channel
	img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	# convert the image to HSV format for color segmentation
	img_hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)

	# mask to extract blue
	lower_blue = np.array([90, 50, 50])
	upper_blue = np.array([120, 240, 240])
	mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

	blue_mask = cv2.bitwise_and(img_output, img_output, mask=mask)

	# seperate out the channels
	r_channel = blue_mask[:, :, 2]
	g_channel = blue_mask[:, :, 1]
	b_channel = blue_mask[:, :, 0]

	# filter out
	filtered_r = cv2.medianBlur(r_channel, 5)
	filtered_g = cv2.medianBlur(g_channel, 5)
	filtered_b = cv2.medianBlur(b_channel, 5)

	# create a blue gray space
	filtered_b = -0.5 * filtered_r + 3 * filtered_b - 2 * filtered_g

	# Do MSER
	regions, _ = mser_blue.detectRegions(np.uint8(filtered_b))
	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

	blank = np.zeros_like(blue_mask)
	cv2.fillPoly(np.uint8(blank), hulls, (255, 0, 0))

	cv2.imshow("Serached", mask)
	cv2.imshow("Result", blank)

	# ADDED MODS TO EXTRACT CONTURS
	_, b_thresh = cv2.threshold(blank[:, :, 0], 60, 255, cv2.THRESH_BINARY)

	cnts, hierarchy = cv2.findContours(b_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	extracted = []
	for i, c in enumerate(cnts):
		M = cv2.moments(c)

		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		x, y, w, h = cv2.boundingRect(c)
		#cv2.rectangle(img, (cX - int(0.8*h), cY - int(0.8*w)), (cX + int(0.8*w), cY + int(0.8*h)), (0, 255, 0), 2)
		cropped_image = img[cY - int(1.2*w):cY + int(1.2*w), cX - int(1.2*h):(cX + int(1.2*w))]
		try:
			resized_image = cv2.resize(cropped_image, (128,128))
			extracted.append(resized_image)
		except:
			pass
	return extracted

def extract_red(imag):

	mser_red = cv2.MSER_create(8, 400, 4000)

	img = imag.copy()
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	# equalize the histogram of the Y channel
	img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	# convert the image to HSV format for color segmentation
	img_hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)

	# mask to extract blue
	lower_red1 = np.array([0, 0, 0])
	upper_red1 = np.array([20, 255, 255])
	mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)

	lower_red2 = np.array([160, 0, 0])
	upper_red2 = np.array([180, 255, 255])
	mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
	mask = mask1 + mask2

	blue_mask = cv2.bitwise_and(img_output, img_output, mask=mask)

	# seperate out the channels
	r_channel = blue_mask[:, :, 2]
	g_channel = blue_mask[:, :, 1]
	b_channel = blue_mask[:, :, 0]

	# filter out
	filtered_r = cv2.medianBlur(r_channel, 5)
	filtered_g = cv2.medianBlur(g_channel, 5)
	filtered_b = cv2.medianBlur(b_channel, 5)

	# create a blue gray space
	filtered_b = -0.5 * filtered_r + 3 * filtered_b - 2 * filtered_g

	# Do MSER
	regions, _ = mser_red.detectRegions(np.uint8(filtered_b))
	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

	blank = np.zeros_like(blue_mask)
	cv2.fillPoly(np.uint8(blank), hulls, (255, 0, 0))

	cv2.imshow("Serached", mask)
	cv2.imshow("Result", blank)

	# ADDED MODS TO EXTRACT CONTURS
	_, b_thresh = cv2.threshold(blank[:, :, 0], 60, 255, cv2.THRESH_BINARY)

	cnts, hierarchy = cv2.findContours(b_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	extracted = []
	for i, c in enumerate(cnts):
		M = cv2.moments(c)

		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		x, y, w, h = cv2.boundingRect(c)
		#cv2.rectangle(img, (cX - int(0.8*h), cY - int(0.8*w)), (cX + int(0.8*w), cY + int(0.8*h)), (0, 255, 0), 2)
		cropped_image = img[cY - int(1.2*w):cY + int(1.2*w), cX - int(1.2*h):(cX + int(1.2*w))]
		try:
			resized_image = cv2.resize(cropped_image, (128,128))
			extracted.append(resized_image)
		except:
			pass
	return extracted


def extract_regions(imag):
	extracted_r = extract_red(imag)
	extracted_b = extract_blue(imag)
	extracted = extracted_r + extracted_b
	return extracted
