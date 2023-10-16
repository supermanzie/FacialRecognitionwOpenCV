import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier(r'haar_face.xml')
people = ['Alex Trebek', 'Doc Rivers']
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'Faces\val\135705789-doc_rivers.png')
#img = cv.imread(r'Faces\val\00Trebeck1-mediumSquareAt3X.jpg')


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 28)

for (x,y,w,h) in faces_rect:
    # roi_gray=gray[y:y+w, x:x+h]
    # label,confidence=face_recognizer.predict(roi_gray)
    faces_roi = gray[y:y+h,x:x+w]
    i, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[i]} with a confidence of {confidence}')
    print(i)
    cv.putText(img, str(people[i]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)