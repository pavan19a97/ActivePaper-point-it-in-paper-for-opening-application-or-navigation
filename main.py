import cv2
import mediapipe as mp
import numpy as np
from pyzbar.pyzbar import decode
import pytesseract
import math

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

roi = [[(566, 726), (846, 896)], [(873, 730), (1150, 910)], [(1160, 733), (1453, 906)], [(1463, 746), (1743, 913)], [(1760, 740), (2016, 893)], [(563, 926), (836, 1096)], [(866, 946), (1160, 1093)], [(1173, 953), (1446, 1093)], [(1470, 940), (1750, 1090)], [(1763, 943), (2030, 1103)], [(573, 1113), (836, 1286)], [(873, 1126), (1143, 1300)], [(1176, 1130), (1440, 1290)], [(1460, 1120), (1730, 1290)], [(1750, 1120), (2033, 1293)], [(543, 1313), (826, 1496)], [(853, 1330), (1130, 1493)], [(1170, 1333), (1436, 1486)], [(1456, 1320), (1730, 1493)], [(1753, 1323), (1993, 1470)], [(563, 1526), (816, 1663)], [(856, 1523), (1130, 1680)], [(1160, 1516), (1443, 1673)], [(1473, 1516), (1746, 1690)], [(1760, 1533), (2010, 1670)]]


def getBarcode(img):
    for barcode in decode(img):
        my_Data = barcode.data.decode('utf-8')
    return my_Data

def getFingerLocation(img):
    width_image, height_image, cupp= img.shape
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # For webcam input:
    hands = mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.4, min_tracking_confidence=0.3, )
    frame = img
    image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    points =[]
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h = hand_landmarks.landmark[8]
            print(h.x, h.y)
            x_px = min(math.floor(h.x * width_image), width_image - 1)
            y_px = min(math.floor(h.y * height_image), height_image - 1)
            points.append(x_px)
            points.append((y_px))

            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', cv2.resize(image, (width_image//4, height_image//4)))
    cv2.waitKey(1)
    print(points)
    return x_px,y_px

def getPaper(img):
    per = 30
    imgQ = cv2.imread('handpaper.jpg')
    imgQ = cv2.resize(imgQ, (1920, 1080))
    h, w, c = imgQ.shape

    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(imgQ, None)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]
    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    try:
        M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
        imgScan = cv2.warpPerspective(img, M, (w, h))
        # imgScaned = cv2.resize(imgScan, (w // 4, h // 4))
        # cv2.imshow("f", imgScaned)
        # cv2.waitKey(1)
        # barCodeData = getBarcode(imgScan)
        # print(barCodeData)
        x_px, y_px = getFingerLocation(img)
        print(x_px, y_px)

        # x_px = points[0]
        # y_px = points[1]
        imgShow = imgScan.copy()
        imgMask = np.zeros_like(imgShow)

        myData = []
        for x, r in enumerate(roi):
            if (x_px>r[0][0] & x_px<r[0][1]) & (y_px>r[0][1] & y_px<r[1][1]  ):
                cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
                imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
                imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
                myData.append(pytesseract.image_to_string(imgCrop))
                cv2.putText(imgShow, str(myData[x]), (r[0][0], r[0][1]),
                            cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 0, 255), 4)
        # imgShow = cv2.resize(imgShow, (w // 4, h // 4))
        cv2.imShow("putText", imgShow)
        cv2.waitKey(1)
        return myData
    except Exception as e:
        print(e)

while cap.isOpened():
    try:
        suc, img = cap.read()
        w,h,c = img.shape
        # img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        # img = cv2.rotate(img, cv2.cv2.ROTATE_180)
        cv2.imshow("orginal as ", cv2.resize(img, (w//4, h//4)))
        cv2.waitKey(1)
        try:

            pointedText = getPaper(img)
            print(pointedText)
            cv2.waitKey(10000)
        except Exception as e:
            print("execption")
            print (e)

        if 0xFFF == "q":
            break
    except:
        print("fucked up")
else:
    cap = cv2.VideoCapture(1)
    cap.set(3, 1920)
    cap.set(4, 1080)
cv2.destroyAllWindows()