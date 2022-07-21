import cv2
import keyboard

from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

from misc import Models, DisplayTitles, Display
from calculations.main import image_model_fitter

cap = cv2.VideoCapture(0)  # 0 - id number for webcam
detector = HandDetector(maxHands=1)
classifier = Classifier(modelPath=Models.MODEL_PATH, labelsPath=Models.LABELS_PATH)

while True:
    success, img = cap.read()
    img_output = img.copy()  # in order not to display unnecessary information to the user
    hands, img = detector.findHands(img=img)

    # crop image
    if hands:
        image_model_fitter(img=img, img_output=img_output, hands=hands, classifier=classifier)

    cv2.imshow(winname=DisplayTitles.MAIN_IMAGE, mat=img_output)
    cv2.waitKey(1)  # 1 ms

    # exit
    if keyboard.is_pressed(Display.EXIT_KEY):
        break
