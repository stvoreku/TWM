import cv2
from skimage.segmentation import slic
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

# image = cv2.imread("detection_test_images/00033.png")
image = cv2.imread("../detection_test_images/pol_09.png")

segments = slic(img_as_float(image), n_segments=100, sigma=5)

fig = plt.figure("Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
plt.axis("off")
plt.show()

# cv2.imshow("Img", image)
# key = cv2.waitKey(0)
