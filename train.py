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

    recs = [0,0]
    precs = [0,0]

    model.train()

    scheduler.step()

    bar = progressbar.ProgressBar(0, len(trainloader), redirect_stdout=False)

    for batch_i, (_, imgs, targets) in enumerate(trainloader):
        imgs = imgs.type(Tensor)
        targets = [x.type(Tensor) for x in targets]

        optimizer.zero_grad()

        loss = model(imgs, targets)
        reg = Tensor([0.0])
        if finetune and indices is None:
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
        recs[0] += model.recprec[0]
        recs[1] += model.recprec[2]
        precs[0] += model.recprec[1]
        precs[1] += model.recprec[3]

        model.seen += imgs.size(0)
    bar.finish()
    prune = count_zero_weights(model)
    print(
        "[Epoch Train %d/%d][Losses: x %f, y %f, w %f, h %f, conf %f, reg %f, pruned %f, total %f, recall: %.5f (%.5f / %.5f), precision: %.5f (%.5f / %.5f)]"
        % (
            epoch + 1,
            epochs,
            lossx / float(len(trainloader)),
            lossy / float(len(trainloader)),
            lossw / float(len(trainloader)),
            lossh / float(len(trainloader)),
            lossconf / float(len(trainloader)),
            lossreg / float(len(trainloader)),
            prune,
            losstotal / float(len(trainloader)),
            recall / float(len(trainloader)),
            recs[0] / float(len(trainloader)),
            recs[1] / float(len(trainloader)),
            prec / float(len(trainloader)),
            precs[0] / float(len(trainloader)),
            precs[1] / float(len(trainloader)),
        )
    )

    name = "DBestFinetune" if finetune else "DBest"
    name = name + ("" if indices is None else "Pruned")

    if bestLoss < (recall + prec):
        bestLoss = (recall + prec)
        model.save_weights("checkpoints/%s.weights" % name)

    return bestLoss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", help="Finetuning",
                        action="store_true")
    parser.add_argument("--lr", help="Learning rate",
                        type=float, default=1e-3)
    parser.add_argument("--decay", help="Weight decay",
                        type=float, default=5e-4)
    opt = parser.parse_args()

    finetune = opt.finetune
    learning_rate = opt.lr
    decay = opt.decay

    classPath = "data/robo.names"
    data_config_path = "config/roboFinetune.data" if finetune else "config/robo.data"
    model_config_path = "config/robo-down-small.cfg"
    img_size = (384,512)
    weights_path = "checkpoints/DBest.weights"
    n_cpu = 4
    batch_size = 64
    epochs = 100 if finetune else 100
    scheduler_step = 50 if finetune else 50

    cuda = torch.cuda.is_available()

    torch.random.manual_seed(12345678)
    if cuda:
        torch.cuda.manual_seed(12345678)


    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    classes = load_classes(classPath)

    # Get data configuration
    data_config = parse_data_config(data_config_path)
    train_path = data_config["train"]
    val_path = data_config["valid"]

    # Get hyper parameters
    hyperparams = parse_model_config(model_config_path)[0]

    # Initiate model
    model = ROBO(model_config_path,img_size=img_size)
    if finetune:
        model.load_weights(weights_path)
    else:
        model.apply(weights_init_normal)

    if cuda:
        model = model.cuda()

    bestLoss = 0

    # Get dataloader
    trainloader = torch.utils.data.DataLoader(
        ListDataset(train_path,img_size=img_size, train=True), batch_size=batch_size, shuffle=True, num_workers=n_cpu
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,50)

    for epoch in range(epochs):
        bestLoss = train(epoch,bestLoss)

    if finetune:
        model.load_weights("checkpoints/DBestFinetune.weights")
        with torch.no_grad():
            indices = pruneModel(model.parameters())

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate*0.1)
        print("Finetuning")

        bestLoss = 0

        for epoch in range(5):
            bestLoss = train(epoch, bestLoss, indices=indices)
