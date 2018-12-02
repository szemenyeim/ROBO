from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='C:/Users/szeme/Documents/projects/UERoboCup/TrainingSetGenerator/Saved/Screenshots/Windows/', help='path to dataset')
    parser.add_argument('--config_path', type=str, default='config/robo-down-small.cfg', help='path to model config file')
    parser.add_argument('--weights_path', type=str, default='checkpoints/90.weights', help='path to weights file')
    parser.add_argument('--class_path', type=str, default='data/robo.names', help='path to class label file')
    parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=(512,640), help='size of each image dimension')
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda if available')
    opt = parser.parse_args()
    print(opt)

    cuda = torch.cuda.is_available() and opt.use_cuda

    os.makedirs('output', exist_ok=True)

    # Set up model
    model = Darknet(opt.config_path, img_size=opt.img_size)
    model.load_weights(opt.weights_path)

    if cuda:
        model.cuda()

    model.eval() # Set in evaluation mode

    dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size, type='%s/*.png'),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    classes = load_classes(opt.class_path) # Extracts class labels from file

    print(classes)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    imgs = []           # Stores image paths
    img_detections = [] # Stores detections for each image index

    print ('\nPerforming object detection:')
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)


        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print ('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)


    print ('\nSaving images:')
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print ("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path).convert('RGB'))

        # The amount of padding that was added
        pad_x = 0
        pad_y = 32
        # Image height and width after padding is removed
        unpad_h = opt.img_size[0] - pad_y
        unpad_w = opt.img_size[1] - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = [(255,0,0),(255,255,0),(0,0,255),(255,0,255)]
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = (y1 - pad_y // 2)
                x1 = (x1 - pad_x // 2) * 0.9
                y2 = (y2 - pad_y // 2)
                x2 = (x2 - pad_x // 2) * 0.9

                color = bbox_colors[int(cls_pred)]
                # Create a Rectangle patch
                cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

        # Save generated image with detections
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        cv2.imwrite('output/%d.png' % (img_i),img)
