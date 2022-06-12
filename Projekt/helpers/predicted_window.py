from tensorflow import nn
import numpy as np
from shapely.geometry import Polygon


class PredictedWindow:
    """
    Holds detection and window data.
    predictions - from Keras
    x,y - image origin (position)
    relative_scale - compared to original image
    """
    def __init__(self, predictions, class_names, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        # self.predictions = predictions  # not needed rn, so save space
        self.softmax_predictions = nn.softmax(predictions[0])
        self.score = np.max(self.softmax_predictions)  # 0..1
        self.predicted_class = class_names[np.argmax(self.softmax_predictions)]

        # Calc rectangle for overlap calculation
        x2 = self.x + self.w
        y2 = self.y + self.h
        self.rectangle = Polygon([
            (x, y),
            (x2, y),
            (x2, y2),
            (x, y2)
        ])


# Returns % of how much do two windows overlap
def get_overlap(a: PredictedWindow, b: PredictedWindow):
    intersection = a.rectangle.intersection(b.rectangle)
    return intersection.area/(a.h*a.w)


def get_worse_window(a: PredictedWindow, b: PredictedWindow):
    if a.score > b.score:
        return b
    else:
        return a
