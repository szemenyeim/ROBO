import glob
import cv2

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

    trPath = "../Data/trafficYolo/Train/"
    trFile = "./data/trafficYolo/train.txt"

    with open(trFile,"w+") as file:
        for fName in glob.glob1(trPath,"*.png"):
            file.write(trPath+fName + "\n")
        file.close()

    '''for fName in glob.glob1(trPath, "*.jpg"):
        print(fName)
        img = cv2.imread(trPath+fName)
        img = image_resize(img,height=416)
        cv2.imwrite(trPath+fName,img)'''

    tePath = "../Data/trafficYolo/Test/"
    teFile = "./data/trafficYolo/test.txt"

    with open(teFile, "w+") as file:
        for fName in glob.glob1(tePath, "*.png"):
            file.write(tePath + fName + "\n")
        file.close()

