import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from operator import itemgetter


def h1_pos(pos):
    global h1
    h1 = pos
    pass


def h2_pos(pos2):
    global h2
    h2 = pos2
    pass


def h3_pos(pos3):
    global h3
    h3 = pos3
    pass


h1 = 8
h2 = 300
h3 = 60

kernel_0 = np.ones((5, 5), np.uint8)

cv2.namedWindow("settings")
cv2.resizeWindow('settings', 300, 300)
cv2.createTrackbar('h1', 'settings', 0, 30, h1_pos)
cv2.createTrackbar('h2', 'settings', 300, 5000, h2_pos)
cv2.createTrackbar('h1', 'settings', 0, 30, h1_pos)
cv2.createTrackbar('h3', 'settings', 60, 255, h3_pos)
kernel = np.ones((3, 3), np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX
imgo = cv2.imread('line4.jpg')

scale_percent = 150  # percent of original size
width = int(imgo.shape[1] * scale_percent / 100)
height = int(imgo.shape[0] * scale_percent / 100)
dim = (width, height)
imgo = cv2.resize(imgo, dim, interpolation=cv2.INTER_AREA)
leftpad = 7500

sp = 25

rightpad = 300
pa = False
allmax = 0
count = 0
rmax = [[0, 0]]
rmin = 300

color = 0
flg = 1

myCXlist = list()
listTemplate = list()
tempSum=list()# [('temp0', temp0)]

# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (700,120))

while (1):
    sttime = time.time()
    timer = cv2.getTickCount()
    k = cv2.waitKey(100) & 0xFF
    if k == 32:
        pa = not pa
    if pa == False:

        leftpad = leftpad - sp
        if leftpad < 0:
            leftpad = 7500
            # count = 0
    img = imgo[:400, int(leftpad):700 + int(leftpad)].copy()
    # img = img[:400,int(leftpad):500+int(leftpad)]
    # cv2.imshow('begin', img)
    # img = cv2.pyrMeanShiftFiltering(img, 21, 51)
    # img = cv2.bilateralFilter(img, 30, 170, 255)
    # cv2.imshow('bil', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.fastNlMeansDenoisingMulti(gray1, 2, 5, None, 4, 7, 35)
    # gray[gray>100]=255
    # cv2.imshow('gray', gray)
    gray = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_TRANSPARENT)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # +cv2.THRESH_OTSU   THRESH_BINARY_INV
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel, iterations=2)
    # opening = cv2.morphologyEx(opening, cv2.MORPH_ELLIPSE, kernel, iterations=2)
    thresh = opening
    opening = cv2.erode(thresh, kernel_0, iterations=3)
    # cv2.imshow('open', opening)

    # kernCircle= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # # kernCircle = np.array([ [0, 0, 0, 0, 0],
    # #                         [0, 0, 3, 0, 0],
    # #                         [0, 3, 3, 3, 0],
    # #                         [0, 0, 3, 0, 0],
    # #                         [0, 0, 0, 0, 0]],dtype=np.uint8)
    # # kernCircle[1,1]=5
    # opened = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernCircle)
    # cv2.imshow('close',opened)
    # dilat = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    #
    # maxd = dist_transform.max()
    # dist_transform[dist_transform<int(maxd/2)]=0
    # if allmax<maxd:
    #     allmax=maxd.copy()
    # mind = dist_transform.min()
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

    # sure_fg = np.uint8(sure_fg)
    # dist_transform = np.uint8(dist_transform)
    # sure_fg= cv2.medianBlur(sure_fg,5)
    # cv2.imshow('sur_fg', sure_fg)
    # unknown = cv2.subtract(dilat, sure_fg)
    unknown2 = cv2.bitwise_not(dist_transform, sure_fg)
    # unknown2[unknown2<-1]=-4
    unknown2 = unknown2 * -10
    unknown2 = np.uint8(unknown2)
    # # cv2.imshow('sur', sure_fg)
    # cv2.normalize(unknown2, unknown2, 0, 255, cv2.NORM_MINMAX)
    # unknown2[unknown2 > h3] = 0
    # unknown2 = cv2.morphologyEx(unknown2, cv2.MORPH_ELLIPSE, kernel, iterations=3)
    # cv2.imshow('unk2', unknown2)
    # cv2.normalize(unknown2, unknown2, 0, 255, cv2.NORM_MINMAX)
    # unknown2=np.uint8(sure_fg.copy())
    #
    # # distr=distr.astype('uint8')
    # #
    # # ret, thresh = cv2.threshold(distr, 200, 255, cv2.THRESH_OTSU)
    # # thresh = cv2.erode(thresh, kernel, iterations=1, borderValue=3)
    # cv2.imshow('sur', sure_fg)
    contours, hierarchy = cv2.findContours(unknown2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)

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
        # cv2.drawContours(img, [cnt], -1, (255, 0, 255),1)
        # cv2.drawContours(sure_fg, [box], -1, (255, 0, 255), 1)
        # cv2.putText(img, str(box[0])+' '+str(box[2]), (box[0],box[1]), font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

        # final = cv2.drawContours(img, contours, contourIdx=-1,color=(255, 0, 0), thickness=2)

        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        if (M['m00'] > h2):


            if cx < rightpad:
                myCXlist.append((cx, cy))
                # if cx>30:
                #     # imCrop=unknown2[cy-30:cy+30,cx-30:cx+30]
                #     imCrop=cv2.getRectSubPix(img,(50,50),(cx,cy))
                #     cv2.imshow('rect1',imCrop)
                #     listTemplate.append([str(len(myCXlist)-1),imCrop])

                cv2.circle(img, (cx, cy), 3, (250, 0, 255), thickness=5)
            # cv2.putText(img, str(rmax), (10, 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 250), 1)
            # cv2.putText(img, str(M['m00']), (cx+10, cy-10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
            # cv2.putText(img, str(cx), (cx - 15, cy - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
            # cv2.putText(img, str(i), (cx - 15, cy - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
            else:
                cv2.circle(img, (cx, cy), 3, (250, 0, 0), thickness=5)

            # cv2.putText(img, str(M['m00']), (cx + 10, cy - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 250), 1)
            # cv2.putText(img, str(count), (10, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 250), 1)


        elif M['m00'] > 0:
            # cv2.putText(img, str(cv2.contourArea(cnt)), (cx + 10, cy - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 250), 1)
            cv2.circle(img, (cx, cy), 3, (0, 0, 255), thickness=5)
        # cv2.putText(img, "#{}".format(int(i) + 1), (int(cx), int(cy) - 15),
        #         cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        # cv2.circle(sure_fg, (cx, cy), 3, (255, 255, 0), thickness=5)
    i = 0
    # if cx<rightpad:
    myCXlist.sort(reverse=True)
    for i in range(0,len(myCXlist)-1):
        imCrop = cv2.getRectSubPix(img, (50, 50), (myCXlist[i][0],myCXlist[i][1]))
        cv2.imshow('rect1', imCrop)
        listTemplate.append([str(i), imCrop])
    listTemplate.clear()

    # while i < len(myCXlist) - 1:
    #     if myCXlist[i + 1][0]+30 >= myCXlist[i][0]:
    #         myCXlist[i + 1] = (myCXlist[i][0], myCXlist[i][1] + 1)
    #         myCXlist.pop(i)
    #     else:
    #         i += 1
    # mx=max(myCXlist)
    # if myCXlist[0][0]>0:
    #     mx=max(myCXlist,key=lambda item:item[1])
    #     if mx[1]>1:
    #         rightpad=270
    #     else:
    #         rightpad = 300
    # #
    # #
    # cv2.putText(img, "#{}".format(time.time() - sttime), (30, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    # print(myCXlist)
    if len(myCXlist) > 0:

        # for i in range(0,len(myCXlist)-1):
        #
        #     cx=myCXlist[i][0]
        #     cy=myCXlist[i][1]
        #     ccx=25; ccy=25
        #     if cx<30:
        #         ccx=21-cx
        #
        #     if cy < 20:
        #         ccy = 20 - cx
        #
        #
        #
        #     if cx>40:
        #         # print(cx, cy, ccx, ccy)
        #         gropp = img[cy-ccy:cy+ccy,cx-ccx:cx+ccx].copy()
        #         listTemplate.append([str(i),gropp])
        #
        #         cv2.imshow(str(i), np.array(listTemplate[i][1]))
        cv2.putText(img, str(i), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # if cx<rmin:
        #     rmin=cx
        # else:
        #     listTemplate = [(str(i), img[cx-40:cy-40,cx+40:cy+40])]
        # if myCXlist[0][1]==2 and myCXlist[0][0]+30>rightpad:
        #     rightpad=350

        if (myCXlist[0][0] >= rmax[0][0]):  # and (myCXlist[0][0] < rightpad)
            rmax[0] = myCXlist[0]
        else:
            count += 1  #rmax[0][1]
            # if myCXlist[0][1]==1:
            #     rightpad=400
            rmax[0] = myCXlist[0]
            # cv2.putText(img, str(rmax[0]), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 150, 255), 2)
    else:
        rmax[0] = [0, 1]

    myCXlist.clear()
    listTemplate.clear()

    img[:, rightpad - 1:rightpad + 1] = (0, 255, 0)
    # img[:,rightpad-301:rightpad-299]=(100,0,100)
    cv2.putText(img, str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 150, 255), 2)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # cv2.putText(img, str(time.time() - sttime), (10, 40),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
    cv2.putText(img, "FPS : " + str(int(fps)), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
    # cv2.putText(img, str(cx), (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

    cv2.imshow('result', img)
    # out.write(img)
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

    # if k != 255:
    #     print(k)
    if k == 27:
        # print(i)
        print(leftpad)
        print(str(' ' + str(time.time() - sttime)), sep='\n', end='')
        # print(M)
        break

# out.release()
cv2.destroyAllWindows()
