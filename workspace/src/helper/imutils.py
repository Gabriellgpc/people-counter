import cv2


def imresize(image, desired_height=None, desired_width=None, keep_aspect=False):
    h, w = image.shape[:2]
    ratio= h/w

    if keep_aspect:
        if desired_height:
            desired_width = int(desired_height * 1.0 / ratio)
        elif desired_width:
            desired_height = int(desired_width * ratio)
    if desired_height != None and desired_width != None:
        image = cv2.resize(image, dsize=[desired_width, desired_height])
    return image