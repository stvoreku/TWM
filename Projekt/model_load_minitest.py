from tensorflow import nn
from tensorflow import expand_dims
import numpy as np
import matplotlib.pyplot as plt

from classifier import *

imfilepath = 'GTSRB-2/Final_Test/Images/000{:02d}.ppm.png'  # formatowanie do 01, 02, etc.
# imfilepath = 'CustomImages/Test/{:02d}.png'
images = []
predictions = []
confidences = []

plt.figure(figsize=(10, 10))

# Te predykcje na pewno można zrobić jakoś grupowo - w przyszłości lepiej nie robić pojedynczo

for i,a in enumerate(range(4,13)):
    fp = imfilepath.format(a)
    img = keras.utils.load_img(fp, target_size=(img_height, img_width))
    img_array = keras.utils.img_to_array(img)
    img_array = expand_dims(img_array, 0)  # Zamiana w tensor, inaczej kształt się nie zgadza

    predictions = model.predict(img_array)
    score = nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    answer_string = "{}\nw/ {:.2f}% confidence".format(predicted_class, 100 * np.max(score))

    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.title(answer_string)
    plt.axis("off")

plt.show()
