import cv2
import numpy as np
import math

from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

from misc import Signs, Display, Models, DisplayTitles

cap = cv2.VideoCapture(0)  # 0 - id number for webcam
detector = HandDetector(maxHands=1)
classifier = Classifier(modelPath=Models.MODEL_PATH, labelsPath=Models.LABELS_PATH)

while True:
    success, img = cap.read()
    img_output = img.copy()  # in order not to display unnecessary information to the user
    hands, img = detector.findHands(img=img)

    # crop image
    if hands:
        hand = hands[0]  # since we have only one hand
        x, y, w, h = hand['bbox']  # bounding box

        # the square in which the img is inscribed
        img_inscribed = np.ones((Display.IMG_SIZE, Display.IMG_SIZE, 3), np.uint8) * 255  # 3 - color

        # dictionary with offsets, where y is height, x is width
        img_crop = img[y - Display.OFFSET:y + h + Display.OFFSET, x - Display.OFFSET:x + w + Display.OFFSET]

        # shifting the image to the white square
        aspect_ratio = h / w  # if > 1 - height is greater than width
        if aspect_ratio > 1:  # stretching by width
            stretch_constant = Display.IMG_SIZE / h  # how much do we need to stretch the image
            width_calculated = math.ceil(stretch_constant * w)  # stretched image in width
            img_resize = cv2.resize(img_crop, (width_calculated, Display.IMG_SIZE))  # resizing image (width, height)
            # centering image
            width_gap = math.ceil((Display.IMG_SIZE - width_calculated) / 2)  # the gap to the center of the image
            # inscribing, [height, width, channel]
            img_inscribed[:, width_gap:width_calculated + width_gap] = img_resize
            # using our model
            prediction, index = classifier.getPrediction(img=img_inscribed, draw=False)
            print(prediction, index)

        else:  # stretching by height
            # similarly
            stretch_constant = Display.IMG_SIZE / w
            height_calculated = math.ceil(stretch_constant * h)
            img_resize = cv2.resize(img_crop, (Display.IMG_SIZE, height_calculated))
            height_gap = math.ceil((Display.IMG_SIZE - height_calculated) / 2)
            img_inscribed[height_gap:height_calculated + height_gap, :] = img_resize
            prediction, index = classifier.getPrediction(img=img_inscribed, draw=False)

        # displaying the information above the hand
        cv2.putText(img=img_output, text=Signs.LABELS[index], org=(x, y - Display.OFFSET * 2),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(143, 254, 9), thickness=2)
        # rectangle around the hand
        cv2.rectangle(img=img_output, pt1=(x - Display.OFFSET, y - Display.OFFSET),
                      pt2=(x + w + Display.OFFSET, y + h + Display.OFFSET), color=(143, 254, 9), thickness=3)

        cv2.imshow(winname=DisplayTitles.CROPPED_IMAGE, mat=img_crop)
        cv2.imshow(winname=DisplayTitles.INSCRIBED_IMAGE, mat=img_inscribed)

    cv2.imshow(winname=DisplayTitles.MAIN_IMAGE, mat=img_output)
    cv2.waitKey(1)  # 1 ms
