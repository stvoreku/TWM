from tensorflow import keras
from tensorflow import nn
import numpy as np
from shapely.geometry import Polygon

print("Loading model...")


img_height = 128  # must be same as in model
img_width = 128
model = keras.models.load_model('cnn_model')

shapes = ['00_circle', '01_triangleUp', '12_priorityFull', '13_yield', '14_stop', '43_nothing']

signs = ['00_speed20', '01_speed30', '02_speed50', '03_speed60', '04_speed70', '05_speed80', '06_endSpeed80',
               '07_speed100', '08_speed120', '09_noPassSmall', '10_noPassBig', '11_prioritySingleIntersection',
               '12_priorityFull', '13_yield', '14_stop', '15_noEntryAll', '16_noEntryLarge', '17_noEntryOneWay',
               '18_dangerGeneric', '19_dangerCurveLeft', '20_dangerCurveRight', '21_dangerDoubleCurveLR',
               '22_dangerRoughRoad', '23_dangerSlippery', '24_dangerNarrowingRight', '25_dangerRoadWorkers',
               '26_dangerTrafficSignal', '27_dangerPedestrians', '28_dangerChildren', '29_dangerBikes',
               '30_dangerIcy', '31_dangerWildlife', '32_endSpeedLimit', '33_mustTurnRight', '34_mustTurnLeft',
               '35_mustDriveForward', '36_mustForwardOrRight', '37_mustForwardOrLeft', '38_mustPassObstacleRight',
               '39_mustPassObstacleLeft', '40_roundabout', '41_endNoPassSmall', '42_endNoPassBig', '43_nothing'
               ]

class_names = shapes

print("Finished.")


class PredictedWindow:
    """
    Holds detection and window data.
    predictions - from Keras
    x,y - image origin (position)
    relative_scale - compared to original image
    """
    def __init__(self, predictions, x, y, relative_scale):
        self.x = x
        self.y = y
        self.w = int(img_width*relative_scale)
        self.h = int(img_height*relative_scale)
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
    return intersection.area/(img_height*img_width)


def get_worse_window(a: PredictedWindow, b: PredictedWindow):
    if a.score > b.score:
        return b
    else:
        return a
