from tensorflow import keras

img_height = 128  # must be same as in model
img_width = 128
model = keras.models.load_model('cnn_model_rgb_signs')

class_names = ['00_speed20', '01_speed30', '02_speed50', '03_speed60', '04_speed70', '05_speed80', '06_endSpeed80',
               '07_speed100', '08_speed120', '09_noPassSmall', '10_noPassBig', '11_prioritySingleIntersection',
               '12_priorityFull', '13_yield', '14_stop', '15_noEntryAll', '16_noEntryLarge', '17_noEntryOneWay',
               '18_dangerGeneric', '19_dangerCurveLeft', '20_dangerCurveRight', '21_dangerDoubleCurveLR',
               '22_dangerRoughRoad', '23_dangerSlippery', '24_dangerNarrowingRight', '25_dangerRoadWorkers',
               '26_dangerTrafficSignal', '27_dangerPedestrians', '28_dangerChildren', '29_dangerBikes',
               '30_dangerIcy', '31_dangerWildlife', '32_endSpeedLimit', '33_mustTurnRight', '34_mustTurnLeft',
               '35_mustDriveForward', '36_mustForwardOrRight', '37_mustForwardOrLeft', '38_mustPassObstacleRight',
               '39_mustPassObstacleLeft', '40_roundabout', '41_endNoPassSmall', '42_endNoPassBig', '43_nothing'
               ]
