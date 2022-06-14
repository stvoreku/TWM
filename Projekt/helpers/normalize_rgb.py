import cv2


def normalize_rgb(img):
    b, g, r = cv2.split(img)

    b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX)

    merge = cv2.merge([b, g, r])
    return merge

# Test
# image = cv2.imread("../detection_test_images/00033.png")
# new_img = normalize_rgb(image)
#
# cv2.imshow("Img", image)
# cv2.imshow("New img", new_img)
# key = cv2.waitKey(0)


