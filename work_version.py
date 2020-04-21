import numpy as np
import cv2
import time

def nothing(pos):
    global h1
    h1=pos
    pass
h1=0

cv2.namedWindow("settings")
cv2.resizeWindow('settings', 300, 300)
cv2.createTrackbar('h1', 'settings', 0, 30, nothing)
kernel = np.ones((3, 3), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
imgo = cv2.imread('IMG_6.jpg')

scale_percent = 30  # percent of original size
width = int(imgo.shape[1] * scale_percent / 100)
height = int(imgo.shape[0] * scale_percent / 100)
dim = (width, height)
imgo = cv2.resize(imgo, dim, interpolation=cv2.INTER_AREA)

while (1):
    img = imgo.copy()
    
    cv2.imshow('begin', img)
    # img = cv2.pyrMeanShiftFiltering(img, 21, 51)
    # img = cv2.bilateralFilter(img, 30, 170, 255)
    # cv2.imshow('bil', img)
    sttime = time.time()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray[gray>100]=255
    cv2.imshow('gray', gray)
    # gray = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_TRANSPARENT)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # +cv2.THRESH_OTSU   THRESH_BINARY_INV
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
    cv2.imshow('open', opening)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # kernCircle= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # # kernCircle = np.array([ [0, 0, 0, 0, 0],
    # #                         [0, 0, 3, 0, 0],
    # #                         [0, 3, 3, 3, 0],
    # #                         [0, 0, 3, 0, 0],
    # #                         [0, 0, 0, 0, 0]],dtype=np.uint8)
    # # kernCircle[1,1]=5
    # opened = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernCircle)
    # cv2.imshow('close',opened)
    dilat = cv2.dilate(opening, kernel, iterations=5)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    maxd = dist_transform.max()
    mind = dist_transform.min()
    ret, sure_fg = cv2.threshold(dist_transform, h1, 255, 0)
    # morph2 = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernCircle)
    # cv2.imshow('dilat', dilat)
    #
    # # distr = cv2.distanceTransform(morph2, cv2.DIST_L2, 5)
    # cv2.normalize(distr, distr, 200, 255, cv2.NORM_MINMAX)
    # distr[distr < 200] = 0
    # # ret, thresh = cv2.threshold(distr, 200, 255, cv2.THRESH_OTSU)
    # distr = np.uint8(distr)
    # # distr[distr < h1] = 0
    # # cv2.normalize(distr, distr, 0, 255, cv2.NORM_MINMAX)
    # dist2 = cv2.bitwise_not(distr, morph)

    sure_fg = np.uint8(sure_fg)
    cv2.imshow('sur_fg', sure_fg)
    unknown = cv2.subtract(dilat, sure_fg)
    unknown2= cv2.bitwise_not(dist_transform,opening)
    cv2.imshow('unk', unknown)
    cv2.imshow('unk2', unknown2)



    #
    # # distr=distr.astype('uint8')
    # #
    # # ret, thresh = cv2.threshold(distr, 200, 255, cv2.THRESH_OTSU)
    # # thresh = cv2.erode(thresh, kernel, iterations=1, borderValue=3)
    # # cv2.imshow('thresh2', thresh)
    contours, hierarchy = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    # print(time.time() - sttime)
    for (i, cnt) in enumerate(contours):
        # ((cX, cY), radius) = cv2.minEnclosingCircle(cnt)
        # cv2.putText(img, "#{}".format(int(i) + 1), (int(cX), int(cY) - 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        # cv2.circle(img, (int(cX), int(cY)), int(radius),
        #            (0, 0, 255), 3)
        # cv2.circle(thresh, (int(cX), int(cY)), int(radius),
        #            (255, 255, 255), 3)
        # count += 1
        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(img, [box], -1, (255, 0, 255),1)
        # cv2.drawContours(sure_fg, [box], -1, (255, 0, 255), 1)
        # cv2.putText(img, str(box[0])+' '+str(box[2]), (box[0],box[1]), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        M = cv2.moments(cnt)
        if (M['m00'] > 0):
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            cv2.circle(img, (cx, cy), 3, (255, 0, 0), thickness=5)
        cv2.putText(img, "#{}".format(int(i) + 1), (int(cx), int(cy) - 15),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    #         cv2.circle(sure_fg, (cx, cy), 3, (255, 255, 0), thickness=5)
    # #
    # #
    # cv2.putText(img, "#{}".format(time.time() - sttime), (30, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    cv2.imshow('result', img)
    # cv2.imshow('result2', sure_fg)
    # res=dilat-distr

    # surefg = cv2.threshold(distr, 0.7 * distr.max(), 255, 0)
    #
    # cv2.imshow('surfg', surefg)
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(dilat, sure_fg)
    #
    #
    # cv2.imshow('substr', unknown)
    # print(time.time()-sttime,end=' ')

    k = cv2.waitKey(1) & 0xFF
    if k != 255:
        print(k)
    if k == 27:
        print(i)
        break
