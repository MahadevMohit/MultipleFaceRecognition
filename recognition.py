import cv2
import train
import detection
import resource


def RecognizeFace(image, faceCascade, eyeCascade, faceSize, threshold):
    found_faces = []

    gray, faces = detection.findFaces(image, faceCascade, eyeCascade, returnGray=1)

    for ((x, y, w, h), eyedim)  in faces:
        label, confidence = recognizer.predict(cv2.resize(detection.levelFace(gray, ((x, y, w, h), eyedim)), faceSize))
        if confidence > threshold:
            found_faces.append((label, confidence, (x, y, w, h)))

    return found_faces



if __name__ == '__main__':
    faceCascade = cv2.CascadeClassifier(resource.FACE_CASCADE)
    eyeCascade = cv2.CascadeClassifier(resource.EYE_CASCADE)
    faceSize = resource.DEFAULT_FACE_SIZE
    threshold = 100

    recognizer = train.trainRecognizer('imgdb', faceSize)

    cv2.namedWindow("camera", 1)
    capture = cv2.VideoCapture(0)

    while True:
        retval, img = capture.read()

        for (label, confidence, (x, y, w, h)) in RecognizeFace(img, faceCascade, eyeCascade, faceSize, threshold):
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, "{} = {}".format(recognizer.getLabelInfo(label), int(confidence)), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1, cv2.LINE_AA)

        cv2.imshow("camera", img)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break