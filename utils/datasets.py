import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize
from skimage.color import rgb2yuv

import sys

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=(512,640), type = '%s/*.*', pad=(16,0)):
        self.files = sorted(glob.glob(type % folder_path))
        self.img_shape = img_size
        self.pad = pad
        self.transform = transforms.Compose([
            transforms.Pad(pad, 128),
            # transforms.Resize(img_size),
            # transforms.ColorJitter(0.2,0.2,0.2,0.2),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].rstrip()
        img = Image.open(img_path).convert('RGB')

        input_img = self.transform(img)

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=(512,640), pad=(16,0)):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = img_size
        self.max_objects = 50
        self.pad = pad
        self.transform = transforms.Compose([
            transforms.Pad(pad,128),
            #transforms.Resize(img_size),
            #transforms.ColorJitter(0.2,0.2,0.2,0.2),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):

        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path).convert('RGB')

        w, h = img.size

        input_img = self.transform(img)

        padded_h, padded_w = h+2*self.pad[0],w+2*self.pad[1]

        #---------
        #  Label
        #---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += self.pad[1]
            y1 += self.pad[0]
            x2 += self.pad[1]
            y2 += self.pad[0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels

    def __len__(self):
        return len(self.img_files)