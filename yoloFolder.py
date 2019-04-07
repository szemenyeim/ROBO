import glob
import cv2
import argparse
import os.path as osp

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Path pointing to the YOLO folder", type=str,required=True)
    opt = parser.parse_args()
    root = opt.root

    trPath = osp.join(root,"YOLO/Train/")
    trFile = "./data/RoboCup/train.txt"

    with open(trFile,"w+") as file:
        for fName in glob.glob1(trPath,"*.png"):
            file.write(trPath+fName + "\n")
        file.close()

    tePath = osp.join(root,"YOLO/Test/")
    teFile = "./data/RoboCup/test.txt"

    with open(teFile, "w+") as file:
        for fName in glob.glob1(tePath, "*.png"):
            file.write(tePath + fName + "\n")
        file.close()

    trPath = osp.join(root,"YOLO/Finetune/train/")
    trFile = "./data/RoboCup/FinetuneTrain.txt"

    with open(trFile,"w+") as file:
        for fName in glob.glob1(trPath,"*.png"):
            file.write(trPath+fName + "\n")
        file.close()

    tePath = osp.join(root,"YOLO/Finetune/test/")
    teFile = "./data/RoboCup/FinetuneTest.txt"

    with open(teFile, "w+") as file:
        for fName in glob.glob1(tePath, "*.png"):
            file.write(tePath + fName + "\n")
        file.close()

