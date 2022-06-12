from tensorflow import keras

img_height = 128  # must be same as in model
img_width = 128
model = keras.models.load_model('cnn_model_shapes_grayscale2')

class_names = ['00_circle', '01_triangleUp', '12_priorityFull', '13_yield', '14_stop', '43_nothing']
