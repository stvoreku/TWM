import cv2
from helpers import normalize_rgb

# Let's load a simple image with 3 black squares
# image = cv2.imread('detection_test_images/pol_01s.png')
# image = cv2.imread('detection_test_images/pol_09.png')
# image = cv2.imread('detection_test_images/pol_12s.png')
# image = cv2.imread('detection_test_images/pol_15.png')
image = cv2.imread("detection_test_images/00033.png")
image = normalize_rgb.normalize_rgb(image)

cv2.waitKey(0)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 30, 200)
cv2.waitKey(0)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# cv2.imshow('Canny Edges After Contouring', edged)
# cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))

boundRect = [None] * len(contours)

for i, con in enumerate(contours):
    contours_poly = cv2.approxPolyDP(con, 3, True)
    boundRect[i] = cv2.boundingRect(contours_poly)

# Draw all contours
# -1 signifies drawing all contours
color = (0, 255, 0)
height_ratio_max = 1.5
width_ratio_max = 1.5
min_side_size = 50

for i, con in enumerate(contours):
    x = int(boundRect[i][0])
    y = int(boundRect[i][1])
    w = boundRect[i][2]
    h = boundRect[i][3]
    if w > min_side_size and h > min_side_size:
        if w/h < width_ratio_max and h/w < height_ratio_max:
            cv2.drawContours(image, con, -1, color, 1)
            cv2.rectangle(image, (x, y), ((x + w), (y + h)), color, 2)
            cv2.imshow('Contours', image)

# cv2.drawContours(image, contours[112], -1, color, 1)
# cv2.imshow('Contours', image)

cv2.waitKey(0)

cv2.destroyAllWindows()
