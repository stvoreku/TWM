import imutils


def scale_pyramid(image, scale_division_step=1.5, steps=1):
    # yield the original image
    yield image

    # keep looping over the pyramid
    for _ in range(steps-1):
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale_division_step)
        image = imutils.resize(image, width=w)

        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]
