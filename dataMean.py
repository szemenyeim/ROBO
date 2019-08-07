import glob
import numpy as np
import cv2

if __name__ == '__main__':
    #root = "E:/RoboCup/FinetuneHorizon/train/images/"
    root = "E:/RoboCup/YOLO/Finetune/train/"
    #root = "E:/RoboCup/train/images/"

    mean = np.zeros(3)
    std = np.zeros(3)

    imgs = glob.glob1(root,"*.png")

    for i in imgs:
        img = cv2.cvtColor(cv2.imread(root+i),cv2.COLOR_BGR2RGB)
        m = np.mean(img,axis=(0,1))
        s = np.sqrt(np.var(img,axis=(0,1)))
        mean += m
        std += s

    mean /= len(imgs)*255
    std /= len(imgs)*255
    std = np.sqrt(std)
    print(mean,std)