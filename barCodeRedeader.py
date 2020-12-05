import cv2
import numpy as np
from pyzbar.pyzbar import decode
# img = cv2.imread('1.png')
cap = cv2.VideoCapture(1)
cap.set(3, 1920)
cap.set(4, 1080)
w= 1920
h = 1080

while True:

    success, img = cap.read()
    img = cv2.rotate(img, cv2.cv2.ROTATE_180)
    cv2.imshow('Orginal', cv2.resize(img,(1920//3,1080//3)))

    cv2.waitKey(1)
    for barcode in decode(img):
        myData = barcode.data.decode('utf-8')
        print(myData)
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 0, 255), 5)
        pts2 = barcode.rect
        cv2.putText(img, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 0, 255), 2)

        cv2.imshow('Orgianal', cv2.resize(img,(w//3, h//3)))
        cv2.waitKey(10)
