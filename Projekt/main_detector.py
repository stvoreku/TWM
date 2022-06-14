
# https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/

import cv2  # opencv-python

import extractor2 as extractor
from tensorflow import keras
from tensorflow import expand_dims

import helpers.classifier_rgb_signs as classifier
from helpers import predicted_window, normalize_rgb
import time

# ------- Display settings ------- #

rect_color = (0, 255, 0)
rect_thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.3
font_color = (255, 0, 255)
font_thickness = 1

# ------- Set up ------- #

image = cv2.imread("detection_test_images/00033.png")
# image = cv2.imread("detection_test_images/pol_09.png")
# display_img = image  # Keep RGB image for display even when using grayscale
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
image = normalize_rgb.normalize_rgb(image)
display_img = image

score_threshold = 0.95
filter_on = False  # filters to leave only high-score detections that aren't 43_nothing
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
        if window.score > 0.95 and window.predicted_class != '43_nothing':
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
for window in chosen_prediction_windows:
    x = window.x
    y = window.y
    w = window.w
    h = window.h

    # text = "{}\nw/ {:.2f}% confidence".format(predicted_class, 100 * np.max(score))
    text = "{:.3f}".format(window.score)
    text2 = window.predicted_class
    text1_pos = (x+rect_thickness, y+rect_thickness)
    text2_pos = (x+rect_thickness, y+h-rect_thickness)

    cv2.rectangle(display_img, (x, y), (x + w, y + h), rect_color, rect_thickness)
    cv2.putText(display_img, text, text1_pos, font, font_scale, font_color, font_thickness)
    cv2.putText(display_img, text2, text2_pos, font, font_scale, font_color, font_thickness)

cv2.imshow("Window", display_img)

# Keep the final image, press Escape to exit
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()
