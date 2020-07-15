import training

import cv2
import pickle
import time
import os

#Face cascade is used to recognize faces in the scene.
#TODO : Use multiple face cascades?
FACE_CASCADE = cv2.CascadeClassifier("Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
#Face recognizer is used to identify faces in the scene.
FACE_RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()

NAMES = {}

curTime = lambda: int(round(time.time() * 1000))

#VideoCapture 0 should default to single webcam. (Multiple webcams may cause issues.)
WEBCAM_STREAM = cv2.VideoCapture(0)

def Idle():
    #Configure the recognizer.
    FACE_RECOGNIZER.read("res\\training_data\\knowledge.yml")
    #Collect and populate the known names.
    #Invert the original names dictionary.
    with open("res\\training_data\\NAME_IDS.pickle", 'rb') as f:
        orig_names = pickle.load(f)
        NAMES = {v: k for k, v in orig_names.items()}

    #Webcam recognition loop.
    while (True):
        #Capture a frame.
        retval, frame = WEBCAM_STREAM.read()
        #Recognize faces in frame.
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(grayFrame)
        for (x, y, w, h) in faces:
            #Identify faces in frame.
            predID, conf = FACE_RECOGNIZER.predict(grayFrame[y:y + h, x:x + w])

            #Display face information on the frame.
            color = (0, 0, 255) #BGR (NOT RGB)
            stroke = 1
            font = cv2.FONT_HERSHEY_PLAIN

            end_coord_x = x + w
            end_coord_y = y + h
            cv2.rectangle(frame, (x,y), (end_coord_x,end_coord_y), color, stroke)
            cv2.putText(frame, "{0} ({1:.2f})".format(NAMES[predID] if (conf < 100) else "Unknown", conf), (x, y), font, 1, (100,100,255), stroke, cv2.LINE_AA)
        #Display the frame.
        cv2.imshow("Webcam Debug Information", frame)

        #Quit Safeguard (Q Key)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #Release resources.
    WEBCAM_STREAM.release()
    cv2.destroyAllWindows()
    return

def CaptureNewTrainingSet(identifier):
    imgIndex = 0
    imgTime = curTime()
    identifier = identifier.replace(" ", "-")

    try:
        os.mkdir("res\\training_data\\{0}".format(identifier))
        print("Creating new training data set for {0}.".format(identifier))
    except OSError as ex:
        print("Updating existing data set for {0}.".format(identifier))

        #Training image ripper loop.
    while (True and imgIndex < 50):
        # Capture a frame.
        retval, frame = WEBCAM_STREAM.read()
        # Recognize faces in frame.
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(grayFrame)
        for (x, y, w, h) in faces:
            color = (0, 0, 255)  # BGR (NOT RGB)
            stroke = 1
            end_coord_x = x + w
            end_coord_y = y + h

            if ((imgTime + 100) <= curTime()):
                training_ROI = grayFrame[y:y + h, x:x + w]
                imgLocation = "res\\training_data\\{0}\\{1}.png".format(identifier, imgIndex)
                cv2.imwrite(imgLocation, training_ROI)
                imgIndex += 1

                color = (0, 255, 0)  # BGR (NOT RGB)
                imgTime = curTime()

            cv2.rectangle(frame, (x, y), (end_coord_x, end_coord_y), color, stroke)

        cv2.imshow("Capturing Training Data", frame)
        cv2.waitKey(1)
    #Release resources.
    WEBCAM_STREAM.release()
    cv2.destroyAllWindows()
    return