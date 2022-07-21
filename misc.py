class Display:
    """
    Main display settings.
    """
    OFFSET = 20
    IMG_SIZE = 300

    FONT_SCALE = 2
    MAIN_COLOR = (143, 254, 9)
    TEXT_THICKNESS = 2
    RECTANGLE_THICKNESS = 3

    SAVE_KEY = 's'
    EXIT_KEY = 'esc'


class Folders:
    """
    Path to main folder and maximum number of images.
    """
    FOLDER_NAME = 'data/'
    MAX_IMAGES_NUMBER = 400


class Signs:
    """
    Signs studied by the model.
    """
    LETTERS = ['A', 'B', 'C', 'D']


class Models:
    """
    Paths to the model and labels
    """
    MODEL_PATH = 'model/keras_model.h5'
    LABELS_PATH = 'model/labels.txt'


class DisplayTitles:
    """
    Titles for displays.
    """
    CROPPED_IMAGE = 'Cropped image'
    INSCRIBED_IMAGE = 'Inscribed image'
    MAIN_IMAGE = 'Main image'
