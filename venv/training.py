import os
import cv2
import pickle

import numpy as np
from PIL import Image

FACE_RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "res\\training_data")

def train():
    print("Training started.")
    CUR_ID = 0
    NAME_IDS = {}
    X_TRAININGS = []
    Y_NAMES = []

    for root, dirs, files in os.walk(TRAINING_DATA_DIR):
        for file in files:
            if file.lower().endswith("png") or file.lower().endswith("jpg"):
                path = os.path.join(root, file)
                name = os.path.basename(os.path.dirname(path))
                if not (name in NAME_IDS):
                    print("Learning {0}...".format(name))
                    NAME_IDS[name] = CUR_ID
                    CUR_ID += 1

                img_array = np.array(Image.open(path))

                X_TRAININGS.append(img_array)
                Y_NAMES.append(NAME_IDS[name])

    print("Training complete.\nUpdating memory.")

    with open("res\\training_data\\NAME_IDS.pickle", 'wb') as f:
        pickle.dump(NAME_IDS, f)

    FACE_RECOGNIZER.train(X_TRAININGS, np.array(Y_NAMES))
    FACE_RECOGNIZER.save("res\\training_data\\knowledge.yml")

    print("Memory updated.")
    return