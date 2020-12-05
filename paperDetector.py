import cv2
import numpy as np
import pytesseract
import os

# C:\Program Files\Tesseract-OCR

per = 30
# pixelThreshold=500

roi = [[(566, 726), (846, 896)], [(873, 730), (1150, 910)], [(1160, 733), (1453, 906)], [(1463, 746), (1743, 913)], [(1760, 740), (2016, 893)], [(563, 926), (836, 1096)], [(866, 946), (1160, 1093)], [(1173, 953), (1446, 1093)], [(1470, 940), (1750, 1090)], [(1763, 943), (2030, 1103)], [(573, 1113), (836, 1286)], [(873, 1126), (1143, 1300)], [(1176, 1130), (1440, 1290)], [(1460, 1120), (1730, 1290)], [(1750, 1120), (2033, 1293)], [(543, 1313), (826, 1496)], [(853, 1330), (1130, 1493)], [(1170, 1333), (1436, 1486)], [(1456, 1320), (1730, 1493)], [(1753, 1323), (1993, 1470)], [(563, 1526), (816, 1663)], [(856, 1523), (1130, 1680)], [(1160, 1516), (1443, 1673)], [(1473, 1516), (1746, 1690)], [(1760, 1533), (2010, 1670)]]


cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread('handpaper.jpg')
h,w,c = imgQ.shape

orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(imgQ,None)
#impKp1 = cv2.drawKeypoints(imgQ,kp1,None)
i=0
while True:
# path = 'resources'
# myPicList = os.listdir(path)
# print(myPicList)
# for j, y in enumerate(myPicList):
#     img = cv2.imread(path + "/" + y)
    suc, img = cap.read()
    imgk = img
    imgk = cv2.resize(imgk, (w // 4, h // 4))

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
        print(M)
        print(type(M))
        imgScan = cv2.warpPerspective(img, M, (w, h))

        imgScaned = cv2.resize(imgScan, (w//4, h//4))
        cv2.imshow("f", imgScaned)



        imgShow = imgScan.copy()
        imgMask = np.zeros_like(imgShow)

        myData = []

        print(f'################## Extracting Data from Form {j}  ##################')

        for x,r in enumerate(roi):

            cv2.rectangle(imgMask, (r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
            # imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
        #
            imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        #     # cv2.imshow(str(x), imgCrop)
        #     print('{} :{}'.format(r[3],pytesseract.image_to_string(imgCrop)))
            myData.append(pytesseract.image_to_string(imgCrop))
            cv2.putText(imgShow,str(myData[x]),(r[0][0],r[0][1]),
                        cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)
        #
        #
        #
        imgShow = cv2.resize(imgShow, (w // 4, h // 4))
        # print(myData)
        cv2.imshow("2", imgShow)
        cv2.waitKey(1)

        cv2.imshow("Camera_input", imgk)
        if cv2.waitKey(1) & 0xFFF == "q":
            break
    except Exception as e :
        i = i + 1
        print(e)
cv2.destroyAllWindows()