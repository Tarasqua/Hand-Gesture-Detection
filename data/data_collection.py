import time
import cv2
import keyboard
import os

from cvzone.HandTrackingModule import HandDetector

from misc import Display, DisplayTitles, Folders
from calculations.main import image_fitter


cap = cv2.VideoCapture(0)  # 0 - id number for webcam
detector = HandDetector(maxHands=1)

FOLDER = ''
NEW_FOLDER = True
COUNTER = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img=img)

    # crop image
    if hands:
        img_inscribed = image_fitter(img=img, hands=hands, img_size=Display.IMG_SIZE, offset=Display.OFFSET)

    cv2.imshow(winname=DisplayTitles.MAIN_IMAGE, mat=img)
    key = cv2.waitKey(1)  # 1 ms

    # saving image (continuous operation)
    if key == ord(Display.SAVE_KEY):
        if NEW_FOLDER:
            FOLDER = Folders.FOLDER_NAME + input("[+] Enter the sign's name: ")
            if not os.path.isdir(FOLDER):
                os.mkdir(FOLDER)
                print(f'[+] Folder {FOLDER} created.')
            NEW_FOLDER = False
        else:
            cv2.imwrite(filename=f'{FOLDER}/image_{time.time()}.png', img=img_inscribed)
            COUNTER += 1
            print(f'[+] {COUNTER}')
            if COUNTER == Folders.MAX_IMAGES_NUMBER:
                print('[+] Completed.')
                time.sleep(1)
                NEW_FOLDER = True
                COUNTER = 0
                continue

    # exit
    if keyboard.is_pressed(Display.EXIT_KEY):
        break
