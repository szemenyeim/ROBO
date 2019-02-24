import glob
import random
import os
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class ImageFolder(Dataset):
    def __init__(self, folder_path, type = '%s/*.*', synth = False):
        self.files = sorted(glob.glob(type % folder_path))
        self.mean = [0.4637419, 0.47166784, 0.48316576] if synth else [0.36224657, 0.41139355, 0.28278301]
        self.std = [0.45211827, 0.16890674, 0.18645908] if synth else [0.3132638, 0.21061972, 0.34144647]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean,std=self.std)
        ])

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].rstrip()
        img = Image.open(img_path)

        input_img = self.transform(img)

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=(384,512), train=True, synth = False):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = img_size
        self.max_objects = 50
        self.train = train
        self.synth = synth
        self.jitter = ColorJitter(0.3,0.3,0.3,3.1415/6)
        self.resize = transforms.Resize(img_size)
        self.mean = [0.4637419, 0.47166784, 0.48316576] if synth else [0.36224657, 0.41139355, 0.28278301]
        self.std = [0.45211827, 0.16890674, 0.18645908] if synth else [0.3132638, 0.21061972, 0.34144647]
        self.normalize = transforms.Normalize(mean=self.mean,std=self.std)

    def __getitem__(self, index):

        #---------
        #  Image
        #---------
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path)

        w, h = img.size

        p = 0
        input_img = transforms.functional.to_tensor(img)
        input_img = self.normalize(input_img)
        if self.train:
            p = torch.rand(1).item()
            if p > 0.5:
                input_img = input_img.flip(2)
            input_img = self.jitter(input_img)

        #---------
        #  Label
        #---------
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        labels = None
        smallLabels = None
        bigLabels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            if p > 0.5:
                labels[:,1] = 1 - labels[:,1]
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            '''x1 += pad[1]
            y1 += pad[0]
            x2 += pad[1]
            y2 += pad[0]
            # Calculate ratios from coordinates'''
            labels[:, 1] = np.clip((((x1 + x2) / 2) / w), a_min=0, a_max = 0.999)
            labels[:, 2] = np.clip((((y1 + y2) / 2) / h), a_min=0, a_max = 0.999)

            smallLabels = np.array([lab for lab in labels if lab[0] < 2])
            bigLabels = np.array([lab for lab in labels if lab[0] >= 2])

        if self.train:
            # Fill matrix
            filled_labels_small = np.zeros((self.max_objects//2, 5))
            filled_labels_big = np.zeros((self.max_objects//2, 5))
            if smallLabels is not None and smallLabels.shape[0] > 0:
                filled_labels_small[range(len(smallLabels))[:self.max_objects]] = smallLabels[:self.max_objects]
            filled_labels_small = torch.from_numpy(filled_labels_small)
            if bigLabels is not None and bigLabels.shape[0] > 0:
                bigLabels[:,0] -= 2
                filled_labels_big[range(len(bigLabels))[:self.max_objects]] = bigLabels[:self.max_objects]
            filled_labels_big = torch.from_numpy(filled_labels_big)

            return img_path, input_img, (filled_labels_small,filled_labels_big)
        else:
            filled_labels = np.zeros((self.max_objects, 5))
            if labels is not None:
                filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
            filled_labels = torch.from_numpy(filled_labels)

            return img_path, input_img, filled_labels


    def __len__(self):
        return len(self.img_files)

def myRGB2YUV(img):
    mtx = torch.FloatTensor([[0.299,0.587,0.114],[-0.14713,-0.28886,0.436],[0.615,-0.51499,-0.10001]])
    return torch.einsum('nm,mbc->nbc',mtx,img)

class ColorJitter(object):
    def __init__(self,b=0.3,c=0.3,s=0.3,h=3.1415/6):
        super(ColorJitter,self).__init__()
        self.b = b
        self.c = c
        self.s = s
        self.h = h

    def __call__(self, img):
        b_val = random.uniform(-self.b,self.b)
        c_val = random.uniform(1-self.c,1+self.c)
        s_val = random.uniform(1-self.s,1+self.s)
        h_val = random.uniform(-self.h,self.h)

        mtx = torch.FloatTensor([[s_val*np.cos(h_val),-np.sin(h_val)],[np.sin(h_val),s_val*np.cos(h_val)]])

        img[0] = (img[0]+b_val)*c_val
        img[1:] = torch.einsum('nm,mbc->nbc',mtx,img[1:])

        return img