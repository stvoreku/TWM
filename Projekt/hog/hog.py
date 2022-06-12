from keras.layers import Layer
from tf_hog import tf_hog_descriptor
from math import ceil, floor


class HOG(Layer):
    def __init__(self,
                 cell_size=8,
                 block_size=2,
                 block_stride=1,
                 n_bins=9,
                 grayscale=True,
                 **kwargs):
        self.cell_size = cell_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.n_bins = n_bins
        self.grayscale = grayscale
        super(HOG, self).__init__(**kwargs)

    def build(self, input_shape):
        super(HOG, self).build(input_shape)

    def call(self, x, *args, **kwargs):
        hog_descriptor = tf_hog_descriptor(x,
                                           self.cell_size,
                                           self.block_size,
                                           self.block_stride,
                                           self.n_bins,
                                           self.grayscale )
        return hog_descriptor

    def compute_output_shape(self, input_shape):
        height, width, channels = input_shape[1:]

        h_cell_count = ceil(height / self.cell_size)
        w_cell_count = ceil(width / self.cell_size)
        h_block_count = \
                floor((h_cell_count - self.block_size + 1) / self.block_stride)
        w_block_count = \
                floor((w_cell_count - self.block_size + 1) / self.block_stride)
        num_dim = h_block_count * w_block_count * self.block_size * \
                self.block_size * self.n_bins
        if not self.grayscale:
            num_dim *= 3
        return input_shape[0], num_dim
