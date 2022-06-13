# import OS module
import os

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# Get the list of all files and directories
path = "GTSRB/Training"
path_to_save = "GTSRB/TrainingHOG"
dir_list = os.listdir(path)

print("Files and directories in '", path, "' :")

for dir in dir_list:
	if dir not in ['.DS_Store', 'Readme.txt']:
		files = os.listdir(f'{path}/{dir}')
		os.mkdir(f"{path_to_save}/{dir}")
		for f in files:
			if 'png' in f:
				img = imread(f'{path}/{dir}/{f}')
				fd, hog_image = hog(img, orientations=9, pixels_per_cell=(4, 4),
									cells_per_block=(2, 2), visualize=True, multichannel=True)
				plt.imsave(f"{path_to_save}/{dir}/{f}", hog_image, cmap="gray")
