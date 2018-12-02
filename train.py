from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import progressbar

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--image_folder", type=str, default="data/RoboCup", help="path to dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch")
    parser.add_argument("--model_config_path", type=str, default="config/robo-down-small.cfg", help="path to model config file")
    parser.add_argument("--data_config_path", type=str, default="config/robo.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/robo.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=(512,640), help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
    )
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
    opt = parser.parse_args()
    print(opt)

    cuda = torch.cuda.is_available() and opt.use_cuda

    torch.random.manual_seed(1234)
    if cuda:
        torch.cuda.manual_seed(1234)


    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    classes = load_classes(opt.class_path)

    # Get data configuration
    data_config = parse_data_config(opt.data_config_path)
    train_path = data_config["train"]

    # Get hyper parameters
    hyperparams = parse_model_config(opt.model_config_path)[0]
    learning_rate = float(hyperparams["learning_rate"])
    momentum = float(hyperparams["momentum"])
    decay = float(hyperparams["decay"])
    burn_in = int(hyperparams["burn_in"])

    # Initiate model
    model = Darknet(opt.model_config_path,img_size=opt.img_size)
    # model.load_weights(opt.weights_path)
    model.apply(weights_init_normal)

    if cuda:
        model = model.cuda()

    model.train()

    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        ListDataset(train_path,img_size=opt.img_size), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,50)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=learning_rate,
                                #momentum=momentum,weight_decay=decay)

    for epoch in range(opt.epochs):
        lossx = 0
        lossy = 0
        lossw = 0
        lossh = 0
        lossconf = 0
        losscls = 0
        losstotal = 0
        recall = 0
        prec = 0

        scheduler.step()

        bar = progressbar.ProgressBar(0, len(dataloader), redirect_stdout=False)

        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            imgs = Variable(imgs.type(Tensor))
            targets = Variable(targets.type(Tensor), requires_grad=False)

            optimizer.zero_grad()

            loss = model(imgs, targets)

            loss.backward()
            optimizer.step()
            bar.update(batch_i)

            '''print(
                "[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
                % (
                    epoch+1,
                    opt.epochs,
                    batch_i+1,
                    len(dataloader),
                    model.losses["x"],
                    model.losses["y"],
                    model.losses["w"],
                    model.losses["h"],
                    model.losses["conf"],
                    model.losses["cls"],
                    loss.item(),
                    model.losses["recall"],
                    model.losses["precision"],
                )
            )'''

            lossx += model.losses["x"]
            lossy += model.losses["y"]
            lossw += model.losses["w"]
            lossh += model.losses["h"]
            lossconf += model.losses["conf"]
            losscls += model.losses["cls"]
            losstotal += loss.item()
            recall += model.losses["recall"]
            prec += model.losses["precision"]

            model.seen += imgs.size(0)
        bar.finish()
        print(
            "[Epoch %d/%d][Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch+1,
                opt.epochs,
                lossx / float(len(dataloader)),
                lossy / float(len(dataloader)),
                lossw / float(len(dataloader)),
                lossh / float(len(dataloader)),
                lossconf / float(len(dataloader)),
                losscls / float(len(dataloader)),
                losstotal / float(len(dataloader)),
                recall / float(len(dataloader)),
                prec / float(len(dataloader)),
            )
        )
        if epoch+1 % opt.checkpoint_interval == 0:
            model.save_weights("%s/%d.weights" % (opt.checkpoint_dir, epoch))
