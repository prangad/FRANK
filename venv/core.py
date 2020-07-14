import cv2
import pickle

#Face cascade is used to recognize faces in the scene.
#TODO : Use multiple face cascades?
FACE_CASCADE = cv2.CascadeClassifier("Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
#Face recognizer is used to identify faces in the scene.
FACE_RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()

FACE_LABELS = {}

#VideoCapture 0 should default to single webcam. (Multiple webcams may cause issues.)
WEBCAM_STREAM = cv2.VideoCapture(0)


def Idle():
    #Configure the recognizer.
    FACE_RECOGNIZER.read("res\\knowledge.yml")
    #Collect and populate the known labels.
    with open("res\\labels.pickle", 'rb') as f:
        invLabels = pickle.load(f)
        # Invert the original labels dictionary.
        FACE_LABELS = {v: k for k, v in invLabels.items()}

    #Webcam recognition loop.
    while (True):
        #Capture a frame.
        retval, frame = WEBCAM_STREAM.read()
        #Recognize faces in frame.
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(grayFrame)
        for (x, y, w, h) in faces:
            #TODO : Identify faces in frame.
            #Display face information on the frame.
            color = (0, 0, 255) #BGR (NOT RGB)
            stroke = 1
            font = cv2.FONT_HERSHEY_PLAIN

            end_coord_x = x + w
            end_coord_y = y + h
            cv2.rectangle(frame, (x,y), (end_coord_x,end_coord_y), color, stroke)
            cv2.putText(frame, "{0} ({1})".format("NAME", "CONF"), (x, y), font, 1, (100,100,255), stroke, cv2.LINE_AA)
        #Display the frame.
        cv2.imshow("Webcam Debug Information", frame)

        #Quit Safeguard (Q Key)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #Release resources.
    wcIn.release()
    cv2.destroyAllWindows()
    return

def CaptureNewTrainingSet(name):
    identifier = 0
    
    #Training image ripper loop.
    while (True):
        break

    #Release resources.
    wcIn.release()
    cv2.destroyAllWindows()
    return