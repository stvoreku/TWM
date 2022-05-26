# Kod pobierający repo w Google Colab:
# !git clone https://github.com/stvoreku/TWM.git

import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers

from tensorflow import keras

from keras.models import Sequential  # Model type to be used
from keras import layers

from keras.utils import np_utils  # NumPy related tools
from keras import optimizers

# ----------------- Config ----------------- #
img_height = 128
img_width = 128
internal_img_size = (img_height, img_width)


def load_dataset(subset="training"):
    return keras.utils.image_dataset_from_directory(
        "GTSRB/Training/",
        labels="inferred",  # labels are generated from the directory structure
        label_mode="int",
        color_mode="rgb",
        batch_size=32,
        image_size=internal_img_size,  # skalowanie obrazu, niestety jest chyba obowiązkowe
        interpolation="bilinear",
        shuffle=True,
        validation_split=0.1,  # Optional float between 0 and 1, fraction of data to reserve for validation
        subset=subset,  # Subset of the data to return; "training" or "validation". Only used if validation_split is set.
        seed=1,  # mandatory for validation_split
        crop_to_aspect_ratio=False  # crop to return the largest possible window from the images without aspect ratio distortion
    )


# ----------------- Main ----------------- #
# https://www.tensorflow.org/tutorials/images/classification

train_dataset = load_dataset()
valid_dataset = load_dataset("validation")

class_names = train_dataset.class_names
print(class_names)

# Ilustracja zbioru, 9 obrazów
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()


# Przykładowa sieć z artykułu
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  # layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Trening
epochs = 10
history = model.fit(
  train_dataset,
  validation_data=valid_dataset,
  epochs=epochs
)

# Wizualizacja treningu

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Zapisywanie modelu
# model.save('cnn_model')

# W Google Colab:
# !zip -r cnn_model.zip cnn_model
# from google.colab import files
# files.download('cnn_model.zip')

# Save class_names

