import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers

from keras.datasets import mnist     # MNIST dataset is included in Keras%
from keras.models import Sequential  # Model type to be used

from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.utils import np_utils                         # NumPy related tools

from tensorflow.keras import optimizers


from sklearn.metrics import confusion_matrix
import itertools


# X - data
# Y - labels
(X_train, y_train), (X_test, y_test) = keras.datasets.image_dataset_from_directory(
    "GTSRB/Training/",
    labels="inferred",  # labels are generated from the directory structure
    label_mode="int",
    color_mode="grayscale",
    batch_size=32,
    image_size=(256, 256),  # to skalowanie chyba niestety jest obowiązkowe
    interpolation="bilinear",
    shuffle=True,
    validation_split=None,  # Optional float between 0 and 1, fraction of data to reserve for validation
    subset=None,  # Subset of the data to return; "training" or "validation". Only used if validation_split is set.
    crop_to_aspect_ratio=False  # crop to return the largest possible window from the images without aspect ratio distortion
)


