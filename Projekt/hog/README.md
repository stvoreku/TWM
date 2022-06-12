# keras-HOG
Differentiable histogram of oriented gradients (HOG) operator implemented in keras using tensorflow as the backend. The implementation is based on [skywind29's tf-HOG](https://github.com/skywind29/tf-HOG) with the following changes:

1. Add a custom [Keras layer](https://keras.io/layers/writing-your-own-keras-layers/) wrapper to the tf-HOG implementation.
2. When computing the gradient magnitude, skywind29 used an [L2 norm](https://github.com/skywind29/tf-HOG/blob/master/tf_hog.py#L51) for `G_x` and `G_y`. Realizing that an L2 norm [causes the derivative to blow up easily](https://github.com/tensorflow/tensorflow/issues/4914). I used an L1 norm here instead. (Please let me know if you think I did it wrong, or you have a better solution)
