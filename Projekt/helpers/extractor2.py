import cv2
import numpy as np

sc = 0.8
min_size = 30

show_masks = False
print_debug = False


def extract_blue(img):
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	lower_blue = np.array([100, 60, 40])
	upper_blue = np.array([130, 255, 200])

	mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

	if show_masks:
		cv2.imshow("Mask blue", mask)

	kernel = np.ones((15, 15), np.uint8)
	mask = cv2.medianBlur(mask, 5)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	contours, hier = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(mask, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)

	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	if show_masks:
		cv2.imshow("Mask blue past morph", mask)

	kernel = np.ones((20, 20), np.uint8)

	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	cnts, hier = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	extracted = []
	regions = []
	for i, c in enumerate(cnts):
		M = cv2.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		x, y, w, h = cv2.boundingRect(c)

		if print_debug:
			print(x, y, w, h)

		if w < min_size and h < min_size:
			continue
		centers = []
		if h / w > 1.5:
			#cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cY1 = cY + h//4
			cY2 = cY - h//4
			h = h//2
			centers.append((cX, cY1))
			centers.append((cX, cY2))
		else:
			centers.append((cX, cY))

		if w > h:
			if print_debug:
				print(w / h)
			w = h
		else:
			if print_debug:
				print(h / w)
			h = w

		if print_debug:
			print(centers)

		# cv2.rectangle(img, (cX - int(0.8*h), cY - int(0.8*w)), (cX + int(0.8*w), cY + int(0.8*h)), (0, 255, 0), 2)
		for center in centers:
			cY = center[1]
			cX = center[0]
			cropped_image = img[cY - int(sc*w):cY + int(sc*w), cX - int(sc*h):(cX + int(sc*w))]
			try:
				resized_image = cv2.resize(cropped_image, (128, 128))
				extracted.append(resized_image)
				regions.append((cX - int(sc*h), cY - int(sc*w), cX + int(sc*h), cY + int(sc*w)))
			except:
				pass
	return extracted, regions


def extract_red(img):
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower_red_1 = np.array([0, 60, 40])
	upper_red_1 = np.array([30, 255, 200])
	mask_1 = cv2.inRange(hsv_img, lower_red_1, upper_red_1)
	lower_red_2 = np.array([170, 60, 40])
	upper_red_2 = np.array([180, 255, 200])
	mask_2 = cv2.inRange(hsv_img, lower_red_2, upper_red_2)
	mask = cv2.bitwise_or(mask_1, mask_2)

	if show_masks:
		cv2.imshow("Mask red", mask)

	kernel = np.ones((15, 15), np.uint8)

	mask = cv2.medianBlur(mask, 3)

	if show_masks:
		cv2.imshow("Mask red past blur", mask)

	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	contours, hier = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	cv2.drawContours(mask, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)

	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	kernel = np.ones((20, 20), np.uint8)

	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

	if show_masks:
		cv2.imshow("Mask red past morph", mask)

	cnts, hier = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	extracted = []
	regions = []
	for i, c in enumerate(cnts):
		M = cv2.moments(c)

		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		x, y, w, h = cv2.boundingRect(c)

		if print_debug:
			print(x, y, w, h)

		if w < min_size and h < min_size:
			continue
		centers = []
		if h / w > 1.5:
			# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cY1 = cY + h // 4
			cY2 = cY - h // 4
			h = h // 2
			centers.append((cX, cY1))
			centers.append((cX, cY2))
		else:
			centers.append((cX, cY))

		if w > h:
			if print_debug:
				print(w / h)
			w = h
		else:
			if print_debug:
				print(h / w)
			h = w

		if print_debug:
			print(centers)
		# cv2.rectangle(img, (cX - int(0.8*h), cY - int(0.8*w)), (cX + int(0.8*w), cY + int(0.8*h)), (0, 255, 0), 2)

		for center in centers:
			cY = center[1]
			cX = center[0]
			cropped_image = img[cY - int(sc * w):cY + int(sc * w), cX - int(sc * h):(cX + int(sc * w))]
			try:
				resized_image = cv2.resize(cropped_image, (128, 128))
				extracted.append(resized_image)
				regions.append((cX - int(sc * h), cY - int(sc * w), cX + int(sc * h), cY + int(sc * w)))
			except:
				pass
	return extracted, regions


def extract_regions(imag):
	extracted_r, regions_r = extract_red(imag)
	extracted_b, regions_b = extract_blue(imag)
	extracted = extracted_r + extracted_b
	regions = regions_b + regions_r
	return extracted, regions
