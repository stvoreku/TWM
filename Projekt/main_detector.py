import cv2  # opencv-python

from tensorflow import keras
from tensorflow import expand_dims

import helpers.classifier_rgb_signs as classifier
from helpers import predicted_window, normalize_rgb, extractor2 as extractor
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ------- Display settings ------- #

rect_color = 'red'
rect_thickness = 1
font_color = 'white'
font_size = 10

# ------- Set up ------- #

extractor.print_debug = False
extractor.show_masks = False

# image = cv2.imread('detection_test_images/pol_01s.png')
# image = cv2.imread('detection_test_images/pol_09.png')
image = cv2.imread('detection_test_images/pol_12s.png')
# image = cv2.imread('detection_test_images/pol_13.png')
# image = cv2.imread('detection_test_images/pol_15.png')
# image = cv2.imread("detection_test_images/00033.png")

image = normalize_rgb.normalize_rgb(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
display_img = image
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

score_threshold = 0.75
filter_on = True  # filters to leave only high-score detections that aren't 43_nothing
remove_overlaps = False

# ------- Main ------- #

prediction_windows = []
print("Processing image...")
start = time.time()

arr, regions = extractor.extract_regions(image)

for i, region in enumerate(regions):
    x = region[0]
    y = region[1]
    w = region[2] - x
    h = region[3] - y
    window = image[y:y + h, x:x + w]
    window = cv2.resize(window, dsize=(128, 128))
    # window = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)  # convert BGR to RGB

    # Classify window using CCN model:
    img_array = keras.utils.img_to_array(window)
    img_array = expand_dims(img_array, 0)
    predictions = classifier.model(img_array)

    pred_win = predicted_window.PredictedWindow(predictions, classifier.class_names, x, y, w, h)
    prediction_windows.append(pred_win)

# print(len(prediction_windows))
print("Finished in", time.time()-start, "seconds.")
print("Processing results...")

# Filter predictions
if filter_on:
    chosen_prediction_windows = []
    for window in prediction_windows:
        if window.score > score_threshold and window.predicted_class != '43_nothing':
            chosen_prediction_windows.append(window)
else:
    chosen_prediction_windows = prediction_windows

# Discard overlapping windows with the same predicted class
if remove_overlaps:
    for window1 in chosen_prediction_windows:
        for window2 in chosen_prediction_windows:
            if window1 != window2 and window1.predicted_class == window2.predicted_class:
                overlap_ratio = predicted_window.get_overlap(window1, window2)
                if overlap_ratio > 0.4:
                    print("Found overlaps of the same class")
                    print(window1.predicted_class, ",", window2.predicted_class, ",", overlap_ratio)
                    tmp = predicted_window.get_worse_window(window1, window2)
                    chosen_prediction_windows.remove(tmp)


print("Drawing results...")

fig, ax = plt.subplots()
ax.imshow(display_img)
for window in chosen_prediction_windows:
    x = window.x
    y = window.y
    w = window.w
    h = window.h

    # text = "{}\nw/ {:.2f}% confidence".format(predicted_class, 100 * np.max(score))
    text = "{:.3f}".format(window.score)
    text2 = window.predicted_class

    rect = patches.Rectangle((x, y), w, h, linewidth=rect_thickness, edgecolor=rect_color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=font_size, color=font_color)
    ax.text(x, y+h, text2, fontsize=font_size, color=font_color)

plt.show()

