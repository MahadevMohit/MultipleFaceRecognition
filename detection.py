import math
import cv2
import numpy as np
import resource


def levelFace(image, face):
    ((x, y, w, h), eyedim) = face
    if len(eyedim) != 2:


        return image[y:y+h, x:x+w]

    leftx = eyedim[0][0]
    lefty = eyedim[0][1]
    rightx = eyedim[1][0]
    righty = eyedim[1][1]
    if leftx > rightx:
        leftx, rightx = rightx, leftx
        lefty, righty = righty, lefty
    if lefty == righty or leftx == rightx:
        return image[y:y+h, x:x+w]

    rotDeg = math.degrees(math.atan((righty - lefty) / float(rightx - leftx)))
    if abs(rotDeg) < 20:
        rotMat = cv2.getRotationMatrix2D((leftx, lefty), rotDeg, 1)
        rotImg = cv2.warpAffine(image, rotMat, (image.shape[1], image.shape[0]))
        return rotImg[y:y+h, x:x+w]

    return image[y:y+h, x:x+w]

def findFaces(image, faceCascade, eyeCascade=None, returnGray=True):
    rejectLevel = 1.3
    levelWeight = 5

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, rejectLevel, levelWeight)

    result = []
    for (x, y, w, h) in faces:
        eyes = []
        if eyeCascade != None:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eyeCascade.detectMultiScale(roi_gray)

        result.append(((x, y, w, h), eyes))

    if returnGray:
        return gray, result
    else:
        return image, result

if __name__ == '__main__':

    width = None
    height = None

    faceCascade = cv2.CascadeClassifier(resource.FACE_CASCADE)
    eyeCascade = cv2.CascadeClassifier(resource.EYE_CASCADE)

    result = np.zeros((height, width, 3), np.uint8)
