import os
import os.path as osp
import glob
import cv2
import random


if __name__ == "__main__":

    inPath = "/Users/martonszemenyei/Projects/ROBO/data/YOLO/Finetune/sydney/"
    oPathTrain = "/Users/martonszemenyei/Projects/ROBO/data/YOLO/Finetune/train/"
    oPathTest = "/Users/martonszemenyei/Projects/ROBO/data/YOLO/Finetune/test/"

    names = sorted(glob.glob1(inPath,"syd*.png"))
    labNames = sorted(glob.glob1(inPath,"*.txt"))

    for img,lab in zip(names,labNames):

        r = random.random()

        if r > 0.8:
            os.rename(osp.join(inPath,img),osp.join(oPathTest,img))
            os.rename(osp.join(inPath,lab),osp.join(oPathTest,lab))
        else:
            os.rename(osp.join(inPath,img),osp.join(oPathTrain,img))
            os.rename(osp.join(inPath,lab),osp.join(oPathTrain,lab))

    '''for name in names:
        img = cv2.imread(oPathTrain+name)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite(oPathTrain+name,img)'''