import cv2
import numpy as np
import pytesseract
import os

# C:\Program Files\Tesseract-OCR

per = 30
# pixelThreshold=500

roi = [[(98, 984), (680, 1074), 'text', 'Name'],
       [(740, 980), (1320, 1078), 'text', 'Phone'],
       [(98, 1154), (150, 1200), 'box', 'Sign'],
       [(738, 1152), (790, 1200), 'box', 'Allergic'],
       [(100, 1418), (686, 1518), 'text', 'Email'],
       [(740, 1416), (1318, 1512), 'text', 'ID'],
       [(110, 1598), (676, 1680), 'text', 'City'],
       [(748, 1592), (1328, 1686), 'text', 'Country']]

cap = cv2.VideoCapture(1)


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files(x86)\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread('rema.jpg')
h,w,c = imgQ.shape
# cv2.imshow("Orginal", cv2.resize(imgQ,(w//3, h//3)))
# cv2.waitKey(0)
#imgQ = cv2.resize(imgQ,(w//3,h//3))

orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
#impKp1 = cv2.drawKeypoints(imgQ,kp1,None)
i=0
# while True:
path = 'resources'
myPicList = os.listdir(path)
print(myPicList)
for j, y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)
    # suc, img = cap.read()
    imgk = img
    imgk = cv2.resize(imgk, (w // 3, h // 3))

    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2,des1)
    matches.sort(key= lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good,None,flags=2)

    # cv2.imshow("hye", imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    try:
        M, _ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
        imgScan = cv2.warpPerspective(img, M, (w, h))
        imgScan = cv2.resize(imgScan, (w//3, h//3))
        cv2.imshow(y, imgScan)
        if cv2.waitKey(0) & 0xFFF == "q":
            break
    except:
        i = i+1
        print(i,"findHomogrraphy error...")

    # imgShow = imgScan.copy()
    # imgMask = np.zeros_like(imgShow)
    #
    # myData = []
    #
    # print(f'################## Extracting Data from Form {j}  ##################')
    #
    # for x,r in enumerate(roi):
    #
    #     cv2.rectangle(imgMask, (r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
    #     imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
    #
    #     imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
    #     # cv2.imshow(str(x), imgCrop)
    #     print('{} :{}'.format(r[3],pytesseract.image_to_string(imgCrop)))
    #     myData.append(pytesseract.image_to_string(imgCrop))
    #     cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]),
    #                 cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)
    #
    #
    #
    # #imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    # print(myData)
    # cv2.imshow("2", imgShow)
    # cv2.imwrite("y", imgShow)
    #
    #
    # #cv2.imshow("KeyPointsQuery",impKp1)
    # cv2.imshow("Output",imgQ)

    cv2.imshow(y+" "+"j", imgk)
    if cv2.waitKey(0) & 0xFFF == "q":
        break
cv2.destroyAllWindows()