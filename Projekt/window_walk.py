
# https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/

import cv2  # opencv-python
from helpers import pyramid, sliding_window

from tensorflow import keras
from tensorflow import expand_dims

import classifier

image = cv2.imread("00033.png")
(winW, winH) = (classifier.img_width, classifier.img_height)

rect_color = (0, 255, 0)
rect_thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.3
font_color = (255, 0, 0)
font_thickness = 1

window_predictions = []
scale_step = 1.5
scale = 1.0  # init

# Loop over the image pyramid
print("Processing image...")
for resized in pyramid(image, scale_division_step=scale_step, steps=3):
    # Loop over the sliding window for each layer of the pyramid
    clone = resized.copy()

    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
        # If the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE WINDOW
        # since we do not have a classifier, we'll just draw the window
        img_array = keras.utils.img_to_array(window)
        img_array = expand_dims(img_array, 0)
        predictions = classifier.model.predict(img_array, verbose=0)
        win_pred = classifier.PredictedWindow(predictions, x, y, scale)
        window_predictions.append(win_pred)

    # Pyramid starts from scale one, so we need to do this at the end of the loop iteration:
    scale /= scale_step

print("Finished.")
print("Drawing results...")

# Draw predictions
for window in window_predictions:

    if window.percent_score > 95 and window.predicted_class != '43_nothing':
        x = window.x
        y = window.y
        w = window.w
        h = window.h

        # text = "{}\nw/ {:.2f}% confidence".format(predicted_class, 100 * np.max(score))
        text = "{:.2f}%".format(window.percent_score)
        text2 = window.predicted_class
        text1_pos = (x+rect_thickness, y+rect_thickness)
        text2_pos = (x+rect_thickness, y+h-rect_thickness)

        cv2.rectangle(image, (x, y), (x + w, y + h), rect_color, rect_thickness)
        cv2.putText(image, text, text1_pos, font, font_scale, font_color, font_thickness)
        cv2.putText(image, text2, text2_pos, font, font_scale, font_color, font_thickness)

cv2.imshow("Window", image)

# Keep the final image, press Escape to exit
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

