import glob
import random
import os
import os.path as osp
import numpy as np

import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numbers
import cv2
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def my_collate(batch):
    imgs,targets,cvimgs = zip(*batch)
    return torch.cat(imgs),torch.cat(targets),cvimgs

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

class RandomAffineCust(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, translate=None, scale=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img, label):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, img.size)

        angle = np.deg2rad(ret[0])
        translations = (ret[1][0]/img.size[0],ret[1][1]/img.size[1])
        scale = ret[2]
        imgRatio = img.size[0]/img.size[1]
        x = (label[:,1]-0.5)*imgRatio
        y = label[:,2]-0.5
        label[:,1] = (x*np.cos(angle) - y*np.sin(angle))*scale/imgRatio + 0.5 + translations[0]
        label[:,2] = (x*np.sin(angle) + y*np.cos(angle))*scale + 0.5 + translations[1]
        label[:, 3] *= scale
        label[:, 4] *= scale

        o_img = F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)
        return o_img, label

class ImageFolder(Dataset):
    def __init__(self, folder_path, type = '%s/*.*', synth = False, yu = False, hr = False):
        self.files = sorted(glob.glob(type % folder_path))
        self.yu = yu
        self.hr = hr
        self.resize = transforms.Resize((192,256))
        self.mean = [0.4637419, 0.47166784, 0.48316576] if synth else [0.36224657, 0.41139355, 0.28278301]
        self.std = [0.45211827, 0.16890674, 0.18645908] if synth else [0.3132638, 0.21061972, 0.34144647]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean,std=self.std)
        ])

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].rstrip()
        img = Image.open(img_path)

        if self.hr:
            img = self.resize(img)

        input_img = self.transform(img)

        if self.yu:
            input_img[1] = input_img[2]*0.5 + input_img[1]*0.5
            input_img = input_img[0:2]

        return img_path, input_img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=(384,512), train=True, synth = False, yu=False):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = img_size
        self.max_objects = 50
        self.train = train
        self.synth = synth
        self.img_size = img_size
        self.yu = yu
        self.jitter = ColorJitter(0.3,0.3,0.3,3.1415/6,0.05)
        self.resize = transforms.Resize(img_size)
        self.affine = RandomAffineCust(5,(0.025,0.025),(0.9,1.1),fillcolor=0)
        self.mean = [0.36269532, 0.41144562, 0.282713] if synth else [0.40513613, 0.48072927, 0.48718367]
        self.std = [0.31111388, 0.21010718, 0.34060917] if synth else [0.44540985, 0.15460468, 0.18062305]
        self.normalize = transforms.Normalize(mean=self.mean,std=self.std)

    def __getitem__(self, index):

        #---------
        #  Image
        #---------
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = Image.open(img_path)

        if self.img_size[0] != img.size[1] and self.img_size[1] != img.size[0]:
            img = self.resize(img)

        w, h = img.size

        # ---------
        #  Label
        # ---------
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        labels = np.loadtxt(label_path).reshape(-1, 5)

        if self.train:
            img,labels = self.affine(img,labels)

        p = 0
        input_img = transforms.functional.to_tensor(img)
        input_img = self.normalize(input_img)
        if self.train:
            p = torch.rand(1).item()
            if p > 0.5:
                input_img = input_img.flip(2)
            input_img = self.jitter(input_img)

        if self.yu:
            input_img[1] = input_img[2]*0.5 + input_img[1]*0.5
            input_img = input_img[0:2]

        if p > 0.5:
            labels[:,1] = 1 - labels[:,1]

        # Squeeze centers inside image
        labels[:, 1] = np.clip(labels[:, 1], a_min=0, a_max = 0.999)
        labels[:, 2] = np.clip(labels[:, 2], a_min=0, a_max = 0.999)

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
    def __init__(self,b=0.3,c=0.3,s=0.3,h=3.1415/6,var=0.05):
        super(ColorJitter,self).__init__()
        self.b = b
        self.c = c
        self.s = s
        self.h = h
        self.var = var

    def __call__(self, img):
        b_val = random.uniform(-self.b,self.b)
        c_val = random.uniform(1-self.c,1+self.c)
        s_val = random.uniform(1-self.s,1+self.s)
        h_val = random.uniform(-self.h,self.h)

        mtx = torch.FloatTensor([[s_val*np.cos(h_val),-np.sin(h_val)],[np.sin(h_val),s_val*np.cos(h_val)]])

        img += torch.randn_like(img)*self.var
        img[0] = (img[0]+b_val)*c_val
        if self.s > 0 and self.h > 0:
            img[1:] = torch.einsum('nm,mbc->nbc',mtx,img[1:])

        return img

class LPDataSet(Dataset):
    def __init__(self, root, img_size=(384,512), train=True, finetune = False, yu=False, len_seq = 2):
        self.finetune = finetune
        self.img_size = img_size
        self.yu = yu
        self.len_seq = len_seq
        self.max_objects = 50
        self.root = osp.join(root,"LabelProp")
        self.split = "train" if train else "val"
        self.resize = transforms.Resize(img_size)
        self.mean = [0.34190056, 0.4833289,  0.48565758] if finetune else [0.36269532, 0.41144562, 0.282713]
        self.std = [0.47421749, 0.13846053, 0.1714848] if finetune else [0.31111388, 0.21010718, 0.34060917]
        self.normalize = transforms.Normalize(mean=self.mean,std=self.std)
        self.images = []
        self.labels = []
        self.predictions = []


        data_dir = osp.join(self.root,"Real" if finetune else "Synthetic")
        data_dir = osp.join(data_dir, self.split)

        for dir in get_immediate_subdirectories(data_dir):
            currDir = osp.join(data_dir,dir)
            img_dir = osp.join(currDir,"images")
            images = []
            for file in sorted(glob.glob1(img_dir, "*.png"), key=alphanum_key):
                images.append(osp.join(img_dir, file))
            self.images.append(images)
            self.labels.append([path.replace('.png', '.txt').replace('.jpg', '.txt') for path in images])

    def __len__(self):
        length = 0
        for imgs in self.images:
            length += len(imgs) - self.len_seq + 1
        return length

    def __getitem__(self, index):
        dirindex = 0
        itemindex = index

        #print index

        for imgs in self.images:
            #print(dirindex, itemindex, len(imgs))
            if itemindex >= len(imgs) - self.len_seq + 1:
                dirindex += 1
                itemindex -= (len(imgs) - self.len_seq + 1)
            else:
                break

        #print(dirindex, itemindex)
        labels = []
        imgs = []
        cvimgs = []
        for i in range(self.len_seq):
            img_file = self.images[dirindex][itemindex+i]
            label_file = self.labels[dirindex][itemindex+i].rstrip()

            img = Image.open(img_file).convert('RGB')
            label = np.loadtxt(label_file).reshape(-1, 5)
            # Squeeze centers inside image
            label[:, 1] = np.clip(label[:, 1], a_min=0, a_max = 0.999)
            label[:, 2] = np.clip(label[:, 2], a_min=0, a_max = 0.999)

            if self.img_size[0] != img.size[1] and self.img_size[1] != img.size[0]:
                img = self.resize(img)

            img_ten = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2YUV)
            img_ten = transforms.functional.to_tensor(img_ten).float()
            img_ten = self.normalize(img_ten)
            if self.yu:
                img_ten[1] = img_ten[2] * 0.5 + img_ten[1] * 0.5
                img_ten = img_ten[0:2]
            img_ten = img_ten.unsqueeze(0)

            filled_label = np.zeros((self.max_objects, 5))
            if label is not None:
                filled_label[range(len(label))[:self.max_objects]] = label[:self.max_objects]
            filled_label = torch.from_numpy(filled_label).unsqueeze(0)

            labels.append(filled_label)
            imgs.append(img_ten)
            cvimgs.append(cv2.resize(cv2.cvtColor(np.array(img),cv2.COLOR_RGB2GRAY),(160,120)))

        imgs = torch.cat(imgs)
        labels = torch.cat(labels)
        return imgs, labels, cvimgs