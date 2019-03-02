from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import argparse

import torch
from torch.utils.data import DataLoader
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

    if indices is None:
        scheduler.step()

    bar = progressbar.ProgressBar(0, len(trainloader), redirect_stdout=False)

    for batch_i, (_, imgs, targets) in enumerate(trainloader):
        imgs = imgs.type(Tensor)
        targets = [x.type(Tensor) for x in targets]

        optimizer.zero_grad()

        loss = model(imgs, targets)
        reg = Tensor([0.0])
        if indices is None:
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

    bar.finish()
    prune = count_zero_weights(model)
    print(
        "[Epoch Train %d/%d lr: %.4f][Losses: x %f, y %f, w %f, h %f, conf %f, reg %f, pruned %f, total %f, recall: %.5f (%.5f / %.5f), precision: %.5f (%.5f / %.5f)]"
        % (
            epoch + 1,
            epochs,
            scheduler.get_lr()[-1]/learning_rate,
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

    name = "bestFinetune" if finetune else "best"
    name +=  "GS" if opt.grayscale else ""
    name +=  "BN" if opt.bn else ""
    if transfer != 0:
        name += "T%d" % transfer
    if indices is not None:
        pruneP = round(prune * 100)
        comp = round(sum(model.get_computations(True))/1000000)
        name = name + ("%d_%d" %(pruneP,comp))

    if bestLoss < (recall + prec):
        bestLoss = (recall + prec)
        torch.save(model.state_dict(), "checkpoints/%s.weights" % name)

    return bestLoss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", help="Finetuning",
                        action="store_true")
    parser.add_argument("--lr", help="Learning rate",
                        type=float, default=1e-3)
    parser.add_argument("--decay", help="Weight decay",
                        type=float, default=1e-4)
    parser.add_argument("--transfer", help="Layers to truly train",
                        action="store_true")
    parser.add_argument("--bn", help="Use bottleneck",
                        action="store_true")
    parser.add_argument("--grayscale", help="Use grayscale images",
                        action="store_true")
    opt = parser.parse_args()

    finetune = opt.finetune
    learning_rate = opt.lr/2 if opt.transfer else opt.lr
    dec = opt.decay if finetune else opt.decay/10
    transfers = ([3, 5, 8, 11] if opt.bn else [3, 5, 7, 9]) if opt.transfer else [0]
    decays = [2e-3, 1e-3, 5e-4, 2.5e-4, 1e-4] if (finetune and not opt.transfer) else [dec]

    classPath = "data/robo.names"
    data_config_path = "config/roboFinetune.data" if finetune else "config/robo.data"
    img_size = (384,512)
    weights_path = "checkpoints/best%s%s.weights" % ("GS" if opt.grayscale else "","BN" if opt.bn else "")
    n_cpu = 4
    batch_size = 64
    channels = 2 if opt.grayscale else 3
    epochs = 125 if opt.transfer == 0 else 150

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    classes = load_classes(classPath)

    # Get data configuration
    data_config = parse_data_config(data_config_path)
    train_path = data_config["train"]
    val_path = data_config["valid"]

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Get dataloader
    trainloader = torch.utils.data.DataLoader(
        ListDataset(train_path,img_size=img_size, train=True, synth=finetune, grayscale=opt.grayscale), batch_size=batch_size, shuffle=True, num_workers=n_cpu
    )

    for transfer in transfers:
        for decay in decays:

            torch.random.manual_seed(1234)
            if cuda:
                torch.cuda.manual_seed(1234)

            # Initiate model
            model = ROBO(inch=channels,bn=opt.bn)
            comp = model.get_computations()
            print(comp)
            print(sum(comp))

            if finetune:
                model.load_state_dict(torch.load(weights_path))

            if cuda:
                model = model.cuda()

            bestLoss = 0

            optimizer = torch.optim.Adam([
                        {'params': model.downPart[0:transfer].parameters(), 'lr': learning_rate*10},
                        {'params': model.downPart[transfer:].parameters()},
                        {'params': model.classifiers.parameters()}
                    ],lr=learning_rate)
            eta_min = learning_rate/25 if opt.transfer else learning_rate/20
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=eta_min)

            for epoch in range(epochs):
                bestLoss = train(epoch,bestLoss)

            if finetune and (transfer == 0):
                model.load_state_dict(torch.load("checkpoints/bestFinetune%s%s.weights" % ("GS" if opt.grayscale else "","BN" if opt.bn else "")))
                with torch.no_grad():
                    indices = pruneModel(model.parameters())

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/40)
                print("Finetuning")

                bestLoss = 0

                for epoch in range(25):
                    bestLoss = train(epoch, bestLoss, indices=indices)
