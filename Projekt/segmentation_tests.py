import cv2
from extractor2 import extract_regions
from helpers.normalize_rgb import normalize_rgb

imag = cv2.imread('detection_test_images/pol_13.png')
cv2.imshow("Before Norm", imag)
imag = normalize_rgb(imag)
arr, regs = extract_regions(imag)

# for i,a in enumerate(arr):
# 	cv2.imwrite(f'contour{i}.png', a)


for reg in regs:
	cv2.rectangle(imag, (reg[0],reg[1]),(reg[2], reg[3]), (0, 255, 0), 2)

cv2.imshow("Conturs", imag)
cv2.waitKey(0)
cv2.destroyAllWindows()
