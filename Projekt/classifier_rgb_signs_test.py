
import helpers.classifier_rgb_signs as classifier
import os
from tensorflow import nn
from tensorflow import expand_dims
import numpy as np
import keras.utils
import matplotlib.pyplot as plt


def ceiling_division(numerator, denominator):
    return -(-numerator // denominator)


imfilepath = 'classification_test_images/ClassTest/'

filenames = os.listdir(imfilepath)

display_data = []

count = len(filenames)
true_pos = 0
false_neg = 0  # didn't detect 43_nothing
false_pos = 0  # detected something but original was 43_nothing
misclassified = 0


for i in range(count):
    fp = imfilepath + filenames[i]

    img = keras.utils.load_img(fp, target_size=(classifier.img_height, classifier.img_width))
    img_array = keras.utils.img_to_array(img)
    img_array = expand_dims(img_array, 0)

    predictions = classifier.model(img_array)
    scores = nn.softmax(predictions[0])
    predicted_class = classifier.class_names[np.argmax(scores)]
    answer_string = "{}\n{:.2f}%".format(predicted_class, 100 * np.max(scores))

    if predicted_class in filenames[i]:
        true_pos += 1
    else:
        # print("Error. Predicted", predicted_class, "for", filenames[i])
        data = [img, answer_string]
        display_data.append(data)

        if '43_nothing' in filenames[i]:
            false_neg += 1
        elif '43_nothing' in predicted_class:
            false_pos += 1
        else:
            misclassified += 1

print('True positive:', true_pos, '/', count)
print('Misclassified:', misclassified, '/', count)
print('False negative:', false_neg, '/', count)
print('False positive:', false_pos, '/', count)
errors = false_neg+false_pos+misclassified
print('Error rate:', errors, '/', count)

print('Displaying detection errors.')

chart_w = 7
chart_h = ceiling_division(errors, chart_w)
# print('chart h:', chart_h)
plt.figure(figsize=(12, 6))  # rozmiar w calach

for i in range(len(display_data)):
    data = display_data[i]
    img = data[0]
    answer_string = data[1]

    ax = plt.subplot(chart_h, chart_w, i + 1)
    plt.imshow(img)
    plt.title(answer_string)
    plt.axis("off")
plt.show()

