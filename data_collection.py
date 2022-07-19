import time
import cv2
import keyboard
import numpy as np
import math

from cvzone.HandTrackingModule import HandDetector

from misc import Display, DisplayTitles, Folders


cap = cv2.VideoCapture(0)  # 0 - id number for webcam
detector = HandDetector(maxHands=1)

NEW_FOLDER = True
COUNTER = 0

while True:
    success, img = cap.read()
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
        else:  # stretching by height
            # similarly
            stretch_constant = Display.IMG_SIZE / w
            height_calculated = math.ceil(stretch_constant * h)
            img_resize = cv2.resize(img_crop, (Display.IMG_SIZE, height_calculated))
            height_gap = math.ceil((Display.IMG_SIZE - height_calculated) / 2)
            img_inscribed[height_gap:height_calculated + height_gap, :] = img_resize

        cv2.imshow(winname=DisplayTitles.CROPPED_IMAGE, mat=img_crop)
        cv2.imshow(winname=DisplayTitles.INSCRIBED_IMAGE, mat=img_inscribed)

    cv2.imshow(winname=DisplayTitles.MAIN_IMAGE, mat=img)
    key = cv2.waitKey(1)  # 1 ms

    # saving image (continuous operation)
    if key == ord('s'):
        if NEW_FOLDER:
            folder = input('[+] Enter the name of the new folder: ')
            NEW_FOLDER = False
        else:
            cv2.imwrite(filename=f'data/{folder.title()}/image_{time.time()}.png', img=img_inscribed)
            COUNTER += 1
            print(f'[+] {COUNTER}')
            if COUNTER == Folders.MAX_IMAGES_NUMBER:
                print('[+] Completed.')
                time.sleep(1)
                NEW_FOLDER = True
                COUNTER = 0
                continue

    # exit
    if keyboard.is_pressed('esc'):
        break
