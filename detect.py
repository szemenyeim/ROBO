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
import progressbar


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, default='checkpoints/DBestFinetunePruned.weights', help='path to weights file')
    parser.add_argument('--class_path', type=str, default='data/robo.names', help='path to class label file')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
    parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
    parser.add_argument('--img_size', type=int, default=(384,512), help='size of each image dimension')
    parser.add_argument("--finetune", help="Finetuning", action="store_true", default=True)
    parser.add_argument("--bn", help="Use bottleneck", action="store_true", default=False)
    parser.add_argument("--yu", help="Use 2 channels", action="store_true", default=False)
    parser.add_argument("--hr", help="Use half res", action="store_true", default=True)
    opt = parser.parse_args()
    print(opt)

    cuda = torch.cuda.is_available()

    image_folder = "E:/RoboCup/YOLO/Finetune/test/" if opt.finetune else "E:/RoboCup/YOLO/Test/"

    weights_path = "checkpoints/bestFinetune" if opt.finetune else "checkpoints/best"

    if opt.yu:
        weights_path += "2C"
    if opt.bn:
        weights_path += "BN"
    if opt.hr:
        weights_path += "HR"

    weights_path += ".weights"

    os.makedirs('output', exist_ok=True)

    # Set up model
    channels = 2 if opt.yu else 3
    model = ROBO(inch=channels,bn=opt.bn,halfRes=opt.hr)
    model.load_state_dict(torch.load(weights_path,map_location={'cuda:0': 'cpu'}))

    print(count_zero_weights(model))

    if cuda:
        model.cuda()

    model.eval() # Set in evaluation mode

    dataloader = DataLoader(ImageFolder(image_folder, synth=opt.finetune, type='%s/*.png', yu=opt.yu, hr=opt.hr),
                            batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    classes = load_classes(opt.class_path) # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    imgs = []           # Stores image paths
    img_detections = [] # Stores detections for each image index

    print ('\nPerforming object detection:')
    bar = progressbar.ProgressBar(0, len(dataloader), redirect_stdout=False)
    elapsed_time = 0
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = input_imgs.type(Tensor)

        # Get detections
        with torch.no_grad():
            start_time = time.time()
            detections = model(input_imgs)
            elapsed_time += time.time() - start_time
            detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)

        # Log progress
        bar.update(batch_i)

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    bar.finish()
    print("\nAverage time: %.2f" % (elapsed_time*1000/len(dataloader)))
    print ('\nSaving images:')
    # Iterate through images and save plot of detections
    bar = progressbar.ProgressBar(0, len(imgs), redirect_stdout=False)
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        # Create plot
        img = np.array(Image.open(path).convert('RGB'))

        # The amount of padding that was added
        pad_x = 0
        pad_y = 0
        # Image height and width after padding is removed
        unpad_h = opt.img_size[0] - pad_y
        unpad_w = opt.img_size[1] - pad_x

        img = cv2.cvtColor(img,cv2.COLOR_YUV2BGR)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = [(0,0,255),(255,0,255),(255,0,0),(0,255,255)]
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = (y1 - pad_y // 2) * 1
                x1 = (x1 - pad_x // 2) * 1
                y2 = (y2 - pad_y // 2) * 1
                x2 = (x2 - pad_x // 2) * 1

                color = bbox_colors[int(cls_pred)]
                # Create a Rectangle patch
                cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

        # Save generated image with detections
        cv2.imwrite('output/%d.png' % (img_i),img)
        bar.update(img_i)
    bar.finish()
