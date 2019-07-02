import cv2
import numpy as np
import os
import os.path as osp
from glob import glob1
import copy
import pickle
import re

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

global sbox, ebox, img, colors, drawing
colors = [(0,0,255),(255,0,255),(255,0,0),(0,255,255)]

def on_mouse(event, x, y, flags, params):
    global sbox, ebox, img, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        sbox = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            ebox = (x, y)
            img2 = img.copy()
            cv2.rectangle(img2,sbox,ebox,colors[classIdx],1)
            cv2.imshow("video", img2)

    elif event == cv2.EVENT_LBUTTONUP:
        ebox = (x, y)
        img2 = img.copy()
        cv2.rectangle(img2,sbox,ebox,colors[classIdx],1)
        cv2.imshow("video", img2)
        drawing = False



if __name__ == '__main__':

    global img, drawing

    drawing = False

    path = "/Users/martonszemenyei/Projects/ROBO/data/YOLO/Finetune/sydney/"

    names = sorted_nicely(glob1(path, "*.png"))

    cv2.namedWindow("video")
    cv2.setMouseCallback("video",on_mouse)

    BBLists = []
    classIdx = 0

    for frameCntr,name in enumerate(names):

        img = cv2.imread(path+name)
        img = cv2.cvtColor(img,cv2.COLOR_YUV2BGR)

        orig = img.copy()
        print(frameCntr)

        if len(BBLists) <= frameCntr:
            BBLists.append([])#copy.deepcopy(BBLists[-1]) if len(BBLists) else [])
        if osp.exists(path + name.split(".")[0] + ".txt"):
            file = open(path + name.split(".")[0] + ".txt", "r")
            BBLists[frameCntr] = []
            while True:
                line = file.readline().split(" ")
                if len(line) < 5:
                    break
                BB = []
                xc = int(float(line[1])*img.shape[1])
                yc = int(float(line[2])*img.shape[0])
                w = int(float(line[3])*img.shape[1])
                h = int(float(line[4])*img.shape[0])
                BB.append((xc-w//2,yc-h//2))
                BB.append((xc+w//2,yc+h//2))
                BB.append(int(line[0]))
                BBLists[frameCntr].append(BB)
        for BB in BBLists[frameCntr]:
            cv2.rectangle(img, BB[0], BB[1], colors[BB[2]], 1)

        BBSel = -1

        BBNum = len(BBLists[frameCntr])

        drawing = False
        cv2.imshow("video", img)

        while True:

            key = cv2.waitKey(20)

            if key == 27:
                exit(0)
            elif key == 13:
                cv2.rectangle(img, sbox, ebox, colors[classIdx], 1)
                cv2.imshow("video", img)
                BBLists[frameCntr].append([sbox, ebox, classIdx])
                BBNum = len(BBLists[frameCntr])
            # k = next image
            elif key == 107:
                classIdx = 0
                BBSel = -1
                break
            # x = del all BBs
            elif key == 120:
                BBLists[frameCntr] = []
                img = orig.copy()
                cv2.imshow("video", img)
            elif key == 48:
                classIdx = 0
            elif key == 49:
                classIdx = 1
            elif key == 50:
                classIdx = 2
            elif key == 51:
                classIdx = 3
            # n = next BB
            elif key == 110:
                if BBNum > 0:
                    BBSel += 1
                    if BBSel == BBNum:
                        BBSel = 0
                    sbox = BBLists[frameCntr][BBSel][0]
                    ebox = BBLists[frameCntr][BBSel][1]
                    img2 = img.copy()
                    cv2.rectangle(img2, sbox, ebox, (255,255,255), 1)
                    cv2.imshow("video", img2)
            # w = widen vertically
            elif key == 119:
                if BBSel >= 0:
                    sbox = BBLists[frameCntr][BBSel][0]
                    ebox = BBLists[frameCntr][BBSel][1]
                    ebox = (ebox[0], ebox[1]+1)
                    BBLists[frameCntr][BBSel][1] = ebox
                    img = orig.copy()
                    for BB in BBLists[frameCntr]:
                        cv2.rectangle(img, BB[0], BB[1], colors[BB[2]], 1)
                    img2 = img.copy()
                    cv2.rectangle(img2, sbox, ebox, (255,255,255), 1)
                    cv2.imshow("video", img2)
            # s = compress vertically
            elif key == 115:
                if BBSel >= 0:
                    sbox = BBLists[frameCntr][BBSel][0]
                    ebox = BBLists[frameCntr][BBSel][1]
                    ebox = (ebox[0], ebox[1]-1)
                    BBLists[frameCntr][BBSel][1] = ebox
                    img = orig.copy()
                    for BB in BBLists[frameCntr]:
                        cv2.rectangle(img, BB[0], BB[1], colors[BB[2]], 1)
                    img2 = img.copy()
                    cv2.rectangle(img2, sbox, ebox, (255,255,255), 1)
                    cv2.imshow("video", img2)
            # a = widen horizontally
            elif key == 97:
                if BBSel >= 0:
                    sbox = BBLists[frameCntr][BBSel][0]
                    ebox = BBLists[frameCntr][BBSel][1]
                    ebox = (ebox[0]+1, ebox[1])
                    BBLists[frameCntr][BBSel][1] = ebox
                    img = orig.copy()
                    for BB in BBLists[frameCntr]:
                        cv2.rectangle(img, BB[0], BB[1], colors[BB[2]], 1)
                    img2 = img.copy()
                    cv2.rectangle(img2, sbox, ebox, (255,255,255), 1)
                    cv2.imshow("video", img2)
            # d = compress horizontally
            elif key == 100:
                if BBSel >= 0:
                    sbox = BBLists[frameCntr][BBSel][0]
                    ebox = BBLists[frameCntr][BBSel][1]
                    ebox = (ebox[0]-1, ebox[1])
                    BBLists[frameCntr][BBSel][1] = ebox
                    img = orig.copy()
                    for BB in BBLists[frameCntr]:
                        cv2.rectangle(img, BB[0], BB[1], colors[BB[2]], 1)
                    img2 = img.copy()
                    cv2.rectangle(img2, sbox, ebox, (255,255,255), 1)
                    cv2.imshow("video", img2)
            # t = move up
            elif key == 116:
                if BBSel >= 0:
                    sbox = BBLists[frameCntr][BBSel][0]
                    ebox = BBLists[frameCntr][BBSel][1]
                    sbox = (sbox[0], sbox[1]+1)
                    BBLists[frameCntr][BBSel][0] = sbox
                    img = orig.copy()
                    for BB in BBLists[frameCntr]:
                        cv2.rectangle(img, BB[0], BB[1], colors[BB[2]], 1)
                    img2 = img.copy()
                    cv2.rectangle(img2, sbox, ebox, (255,255,255), 1)
                    cv2.imshow("video", img2)
            # g = move down
            elif key == 103:
                if BBSel >= 0:
                    sbox = BBLists[frameCntr][BBSel][0]
                    ebox = BBLists[frameCntr][BBSel][1]
                    sbox = (sbox[0], sbox[1]-1)
                    BBLists[frameCntr][BBSel][0] = sbox
                    img = orig.copy()
                    for BB in BBLists[frameCntr]:
                        cv2.rectangle(img, BB[0], BB[1], colors[BB[2]], 1)
                    img2 = img.copy()
                    cv2.rectangle(img2, sbox, ebox, (255,255,255), 1)
                    cv2.imshow("video", img2)
            # f = move left
            elif key == 102:
                if BBSel >= 0:
                    sbox = BBLists[frameCntr][BBSel][0]
                    ebox = BBLists[frameCntr][BBSel][1]
                    sbox = (sbox[0]+1, sbox[1])
                    BBLists[frameCntr][BBSel][0] = sbox
                    img = orig.copy()
                    for BB in BBLists[frameCntr]:
                        cv2.rectangle(img, BB[0], BB[1], colors[BB[2]], 1)
                    img2 = img.copy()
                    cv2.rectangle(img2, sbox, ebox, (255,255,255), 1)
                    cv2.imshow("video", img2)
            # h = move right
            elif key == 104:
                if BBSel >= 0:
                    sbox = BBLists[frameCntr][BBSel][0]
                    ebox = BBLists[frameCntr][BBSel][1]
                    sbox = (sbox[0]-1, sbox[1])
                    BBLists[frameCntr][BBSel][0] = sbox
                    img = orig.copy()
                    for BB in BBLists[frameCntr]:
                        cv2.rectangle(img, BB[0], BB[1], colors[BB[2]], 1)
                    img2 = img.copy()
                    cv2.rectangle(img2, sbox, ebox, (255,255,255), 1)
                    cv2.imshow("video", img2)
            # r = remove BB
            elif key == 114:
                if BBSel >= 0:
                    BBLists[frameCntr].pop(BBSel)
                    BBSel = -1
                    img = orig.copy()
                    for BB in BBLists[frameCntr]:
                        cv2.rectangle(img, BB[0], BB[1], colors[BB[2]], 1)
                    cv2.imshow("video", img)
                    BBNum = len(BBLists[frameCntr])

        file = open(path + name.split(".")[0] + ".txt","w")
        for BB in BBLists[frameCntr]:
            center = ((BB[0][0] + BB[1][0])/(2*img.shape[1]), (BB[0][1] + BB[1][1])/(2*img.shape[0]))
            size = (abs(BB[1][0] - BB[0][0]) / img.shape[1],abs(BB[1][1] - BB[0][1]) / img.shape[0])
            label = BB[2]
            file.write(str(label))
            file.write(" ")
            file.write(str(center[0]))
            file.write(" ")
            file.write(str(center[1]))
            file.write(" ")
            file.write(str(size[0]))
            file.write(" ")
            file.write(str(size[1]))
            file.write("\n")
        file.close()