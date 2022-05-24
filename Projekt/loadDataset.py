import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers

from tensorflow import keras

from keras.models import Sequential  # Model type to be used
from keras.layers.core import Dense, Dropout, Activation  # Types of layers to be used in our model

from keras.utils import np_utils                         # NumPy related tools
from keras import optimizers


def load_dataset(subset="training"):
    return keras.utils.image_dataset_from_directory(
        "GTSRB/Training/",
        labels="inferred",  # labels are generated from the directory structure
        label_mode="int",
        color_mode="grayscale",
        batch_size=32,
        image_size=(256, 256),  # to skalowanie chyba niestety jest obowiązkowe
        interpolation="bilinear",
        shuffle=True,
        validation_split=0.99,  # Optional float between 0 and 1, fraction of data to reserve for validation
        subset=subset,  # Subset of the data to return; "training" or "validation". Only used if validation_split is set.
        seed=1,  # mandatory for validation_split
        crop_to_aspect_ratio=False  # crop to return the largest possible window from the images without aspect ratio distortion
    )


train_dataset = load_dataset()
# valid_dataset = load_dataset("validation")

# https://www.tensorflow.org/tutorials/images/classification

class_names = train_dataset.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()


