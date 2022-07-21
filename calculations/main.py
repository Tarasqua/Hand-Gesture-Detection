import cv2
import numpy as np
import math

from misc import Display, DisplayTitles, Signs
from cvzone.ClassificationModule import Classifier


def image_fitter(img: None, hands: list, img_size: int = Display.IMG_SIZE, offset: int = Display.OFFSET,
                 title_cropped_image: str = DisplayTitles.CROPPED_IMAGE,
                 title_inscribed_image: str = DisplayTitles.INSCRIBED_IMAGE):
    """
    Crops and fits the image into the white square.
    :param img: the original image.
    :param hands: hands on the image.
    :param img_size: constant image size (optional).
    :param offset: constant offset (optional).
    :param title_cropped_image: constant cropped image title (optional).
    :param title_inscribed_image: constant inscribed image title (optional).
    :return: img_inscribed - cropped and fitted image into the white square.
    """
    hand = hands[0]  # since we have only one hand
    x, y, w, h = hand['bbox']  # bounding box

    # the square in which the img is inscribed
    img_inscribed = np.ones((img_size, img_size, 3), np.uint8) * 255  # 3 - color

    # dictionary with offsets, where y is height, x is width
    img_crop = img[y - offset:y + h + offset, x - offset:x + w + offset]

    aspect_ratio = h / w  # if > 1 - height is greater than width
    if aspect_ratio > 1:  # stretching by width
        stretch_constant = img_size / h  # how much do we need to stretch the image
        width_calculated = math.ceil(stretch_constant * w)  # stretched image in width
        img_resize = cv2.resize(img_crop, (width_calculated, img_size))  # resizing image (width, height)
        # centering image
        width_gap = math.ceil((img_size - width_calculated) / 2)  # the gap to the center of the image
        # inscribing, [height, width, channel]
        img_inscribed[:, width_gap:width_calculated + width_gap] = img_resize

    else:  # stretching by height
        # similarly
        stretch_constant = img_size / w
        height_calculated = math.ceil(stretch_constant * h)
        img_resize = cv2.resize(img_crop, (img_size, height_calculated))
        height_gap = math.ceil((img_size - height_calculated) / 2)
        img_inscribed[height_gap:height_calculated + height_gap, :] = img_resize

    cv2.imshow(winname=title_cropped_image, mat=img_crop)
    cv2.imshow(winname=title_inscribed_image, mat=img_inscribed)

    return img_inscribed


def image_model_fitter(img: None, img_output: None, hands: list, classifier: Classifier, letters: list = Signs.LETTERS,
                       img_size: int = Display.IMG_SIZE, offset: int = Display.OFFSET,
                       title_cropped_image: str = DisplayTitles.CROPPED_IMAGE,
                       title_inscribed_image: str = DisplayTitles.INSCRIBED_IMAGE):
    """
    Crops and fits the image into the white square and, using the learning model,
    outputs the necessary information - which sign is shown.
    :param img: the original image.
    :param img_output: a copy of the original image.
    :param hands: hands on the image.
    :param classifier: Classifier class object with the model used.
    :param letters: signs (alphabet letters) used in the model (optional)
    :param img_size: constant image size (optional).
    :param offset: constant offset (optional).
    :param title_cropped_image: constant cropped image title (optional).
    :param title_inscribed_image: constant inscribed image title (optional).
    :return: void function
    """

    hand = hands[0]  # since we have only one hand
    x, y, w, h = hand['bbox']  # bounding box

    # the square in which the img is inscribed
    img_inscribed = np.ones((img_size, img_size, 3), np.uint8) * 255  # 3 - color

    # dictionary with offsets, where y is height, x is width
    img_crop = img[y - offset:y + h + offset, x - offset:x + w + offset]

    # shifting the image to the white square
    aspect_ratio = h / w  # if > 1 - height is greater than width
    if aspect_ratio > 1:  # stretching by width
        stretch_constant = img_size / h  # how much do we need to stretch the image
        width_calculated = math.ceil(stretch_constant * w)  # stretched image in width
        img_resize = cv2.resize(img_crop, (width_calculated, img_size))  # resizing image (width, height)
        # centering image
        width_gap = math.ceil((img_size - width_calculated) / 2)  # the gap to the center of the image
        # inscribing, [height, width, channel]
        img_inscribed[:, width_gap:width_calculated + width_gap] = img_resize
        # using our model
        prediction, index = classifier.getPrediction(img=img_inscribed, draw=False)
        print(prediction, index)

    else:  # stretching by height
        # similarly
        stretch_constant = img_size / w
        height_calculated = math.ceil(stretch_constant * h)
        img_resize = cv2.resize(img_crop, (img_size, height_calculated))
        height_gap = math.ceil((img_size - height_calculated) / 2)
        img_inscribed[height_gap:height_calculated + height_gap, :] = img_resize
        prediction, index = classifier.getPrediction(img=img_inscribed, draw=False)

    # displaying the information above the hand
    cv2.putText(img=img_output, text=letters[index], org=(x, y - offset * 2),
                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(143, 254, 9), thickness=2)
    # rectangle around the hand
    cv2.rectangle(img=img_output, pt1=(x - offset, y - offset),
                  pt2=(x + w + offset, y + h + offset), color=(143, 254, 9), thickness=3)

    cv2.imshow(winname=title_cropped_image, mat=img_crop)
    cv2.imshow(winname=title_inscribed_image, mat=img_inscribed)
