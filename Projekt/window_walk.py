
# https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/

import cv2  # opencv-python
from helpers import pyramid, sliding_window

from tensorflow import keras
from tensorflow import expand_dims

import classifier
import time

image = cv2.imread("00033.png")
display_img = image
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
(winW, winH) = (classifier.img_width, classifier.img_height)

rect_color = (0, 255, 0)
rect_thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.3
font_color = (255, 0, 255)
font_thickness = 1

prediction_windows = []
scale_step = 1.5
scale = 1.0  # init

# Loop over the image pyramid
print("Processing image...")
start = time.time()
for resized in pyramid(image, scale_division_step=scale_step, steps=4):
    # Loop over the sliding window for each layer of the pyramid
    clone = resized.copy()

    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
        # If the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # Classify window using CCN model:
        img_array = keras.utils.img_to_array(window)
        img_array = expand_dims(img_array, 0)
        predictions = classifier.model(img_array)
        pred_win = classifier.PredictedWindow(predictions, x, y, scale)
        prediction_windows.append(pred_win)

    # Pyramid starts from scale one, so we need to do this at the end of the loop iteration:
    scale /= scale_step

print("Finished in", time.time()-start, "seconds.")
print("Processing results...")

# Filter predictions
chosen_prediction_windows = []
for window in prediction_windows:
    if window.score > 0.95 and window.predicted_class != '43_nothing':
        chosen_prediction_windows.append(window)

# Discard overlapping windows with the same predicted class
for window1 in chosen_prediction_windows:
    for window2 in chosen_prediction_windows:
        if window1 != window2 and window1.predicted_class == window2.predicted_class:
            overlap_ratio = classifier.get_overlap(window1, window2)
            if overlap_ratio > 0.4:
                print("Found overlaps of the same class")
                print(window1.predicted_class, ",", window2.predicted_class, ",", overlap_ratio)
                tmp = classifier.get_worse_window(window1, window2)
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

