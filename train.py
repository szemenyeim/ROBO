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

def add_dimension_glasso(var, dim=0):
    return var.pow(2).sum(dim=dim).add(1e-8).pow(1/2.).sum()

def get_glasso_term(model):
    loss = 0
    for param in model.parameters():
        if param.data.dim() == 4:
            loss += add_dimension_glasso(param,0)
            loss += add_dimension_glasso(param,1)
    return loss

def l1reg(model):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(torch.abs(param))
    return regularization_loss

def train(epoch,bestLoss, indices = None):
    #############
    ####TRAIN####
    #############

    lossx = 0
    lossy = 0
    lossw = 0
    lossh = 0
    lossconf = 0
    lossreg = 0
    losstotal = 0
    recall = 0
    prec = 0

    model.train()

    scheduler.step()

    bar = progressbar.ProgressBar(0, len(trainloader), redirect_stdout=False)

    for batch_i, (_, imgs, targets) in enumerate(trainloader):
        imgs = imgs.type(Tensor)
        targets = targets.type(Tensor)

        optimizer.zero_grad()

        loss = model(imgs, targets)
        reg = decay * l1reg(model)
        loss += reg

        loss.backward()

        if indices is not None:
            pIdx = 0
            for param in model.parameters():
                if param.dim() > 1:
                    if param.grad is not None:
                        param.grad[indices[pIdx]] = 0
                    pIdx += 1

        optimizer.step()
        bar.update(batch_i)

        lossx += model.losses["x"]
        lossy += model.losses["y"]
        lossw += model.losses["w"]
        lossh += model.losses["h"]
        lossconf += model.losses["conf"]
        lossreg += reg.item()
        losstotal += loss.item()
        recall += model.losses["recall"]
        prec += model.losses["precision"]

        model.seen += imgs.size(0)
    bar.finish()
    prune = count_zero_weights(model)
    print(
        "[Epoch Train %d/%d][Losses: x %f, y %f, w %f, h %f, conf %f, reg %f, pruned %f, total %f, recall: %.5f, precision: %.5f]"
        % (
            epoch + 1,
            opt.epochs,
            lossx / float(len(trainloader)),
            lossy / float(len(trainloader)),
            lossw / float(len(trainloader)),
            lossh / float(len(trainloader)),
            lossconf / float(len(trainloader)),
            lossreg / float(len(trainloader)),
            prune,
            losstotal / float(len(trainloader)),
            recall / float(len(trainloader)),
            prec / float(len(trainloader)),
        )
    )

    name = "best" if indices is None else "pruned"

    if bestLoss < (recall + prec):
        bestLoss = (recall + prec)
        model.save_weights("%s/%s.weights" % (opt.checkpoint_dir,name))

    return bestLoss

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
    parser.add_argument("--img_size", type=int, default=(384,512), help="size of each image dimension")
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
    val_path = data_config["valid"]

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
    print(count_zero_weights(model))

    if cuda:
        model = model.cuda()

    bestLoss = 0

    # Get dataloader
    trainloader = torch.utils.data.DataLoader(
        ListDataset(train_path,img_size=opt.img_size), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu
    )
    valLoader = torch.utils.data.DataLoader(
        ListDataset(val_path,img_size=opt.img_size), batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer = torch.optim.Adam(model.parameters())
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=learning_rate,momentum=momentum,weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,50)

    for epoch in range(opt.epochs):
        bestLoss = train(epoch,bestLoss)


        #############
        #####VAL#####
        #############
        '''lossx = 0
        lossy = 0
        lossw = 0
        lossh = 0
        lossconf = 0
        lossreg = 0
        losstotal = 0
        recall = 0
        prec = 0

        model.eval()

        bar = progressbar.ProgressBar(0, len(valLoader), redirect_stdout=False)

        for batch_i, (_, imgs, targets) in enumerate(valLoader):
            with torch.no_grad():
                imgs = imgs.type(Tensor)
                targets = targets.type(Tensor)

                loss = model(imgs, targets)
                reg = decay*get_glasso_term(model)
                loss += reg

                bar.update(batch_i)

                lossx += model.losses["x"]
                lossy += model.losses["y"]
                lossw += model.losses["w"]
                lossh += model.losses["h"]
                lossconf += model.losses["conf"]
                lossreg += reg.item()
                losstotal += loss.item()
                recall += model.losses["recall"]
                prec += model.losses["precision"]

        bar.finish()
        print(
            "[Epoch Val %d/%d][Losses: x %f, y %f, w %f, h %f, conf %f, reg %f, total %f, recall: %.5f, precision: %.5f]"
            % (
                epoch + 1,
                opt.epochs,
                lossx / float(len(valLoader)),
                lossy / float(len(valLoader)),
                lossw / float(len(valLoader)),
                lossh / float(len(valLoader)),
                lossconf / float(len(valLoader)),
                lossreg / float(len(valLoader)),
                losstotal / float(len(valLoader)),
                recall / float(len(valLoader)),
                prec / float(len(valLoader)),
            )
        )'''

    model.load_weights("%s/best.weights" % opt.checkpoint_dir)
    with torch.no_grad():
        indices = pruneModel(model.parameters())

    print("Finetuning")

    bestLoss = 0

    for epoch in range(10):
        bestLoss = train(epoch, bestLoss, indices)
