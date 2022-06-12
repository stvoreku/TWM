
# Used to generate non-sign dataset

import cv2  # opencv-python
from window_slider import sliding_window
import os
from classifier import img_height, img_width


(winW, winH) = (img_width, img_height)
scale_step = 1.5
path_of_the_directory = 'neg_training_full_pics'

for filename in os.listdir(path_of_the_directory):
    f = os.path.join(path_of_the_directory, filename)
    image = cv2.imread(f)
    print(f)
    i = 0
    for (x, y, window) in sliding_window(image, stepSize=winW, windowSize=(winW, winH)):
        # If the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        f_name = filename + str(i) + ".png"
        filepath = os.path.join("../GTSRB/Training/43_nothing", f_name)
        print(filepath)
        cv2.imwrite(filepath, window)
        i += 1
