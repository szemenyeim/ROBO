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

def train(epoch,epochs,bestLoss,indices = None):
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
    name +=  "2C" if opt.yu else ""
    name +=  "BN" if opt.bn else ""
    name +=  "HR" if opt.hr else ""
    if transfer != 0:
        name += "T%d" % transfer
    if indices is not None:
        pruneP = round(prune * 100)
        comp = round(sum(model.get_computations(True))/1000000)
        name = name + ("%d_%d" %(pruneP,comp))

    '''if bestLoss < (recall + prec):
        print("Saving best model")
        bestLoss = (recall + prec)
        torch.save(model.state_dict(), "checkpoints/%s.weights" % name)'''

    return bestLoss


def valid(epoch,epochs,bestLoss,pruned):
    #############
    ####VALID####
    #############

    model.eval()

    mAP, APs = computeAP(model,valloader,0.5,0.45,4,(384,512),False,32)
    prune = count_zero_weights(model)

    name = "bestFinetune" if finetune else "best"
    name +=  "2C" if opt.yu else ""
    name +=  "BN" if opt.bn else ""
    name +=  "HR" if opt.hr else ""
    if transfer != 0:
        name += "T%d" % transfer
    if pruned:
        pruneP = round(prune * 100)
        comp = round(sum(model.get_computations(True))/1000000)
        name = name + ("%d_%d" %(pruneP,comp))

    print("[Epoch Val %d/%d mAP: %.4f][Ball: %.4f Crossing: %.4f Goalpost: %.4f Robot: %.4f]" % (epoch + 1,epochs,mAP,APs[0],APs[1],APs[2],APs[3]))

    if bestLoss < (mAP):
        print("Saving best model")
        bestLoss = (mAP)
        torch.save(model.state_dict(), "checkpoints/%s.weights" % name)

    return bestLoss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", help="Finetuning", action="store_true", default=False)
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    parser.add_argument("--decay", help="Weight decay", type=float, default=1e-4)
    parser.add_argument("--transfer", help="Layers to truly train", action="store_true")
    parser.add_argument("--bn", help="Use bottleneck", action="store_true")
    parser.add_argument("--yu", help="Use 2 channels", action="store_true", default=False)
    parser.add_argument("--hr", help="Use half res", action="store_true", default=False)
    parser.add_argument("--singleDec", help="Just use a single decay value", action="store_true", default=False)
    opt = parser.parse_args()

    finetune = opt.finetune
    learning_rate = opt.lr/2 if opt.transfer else opt.lr
    dec = opt.decay if finetune else opt.decay/10
    transfers = ([3, 5, 8, 11] if opt.bn else [3, 5, 7, 9]) if opt.transfer else [0]
    decays = [dec*100, dec*25, dec*10, dec*2.5, dec] if (finetune and not opt.transfer) else [dec]
    if opt.singleDec:
        decays = [dec*20]
    halfRes = opt.hr

    classPath = "data/robo.names"
    data_config_path = "config/roboFinetune.data" if finetune else "config/robo.data"
    img_size = (192,256) if halfRes else (384,512)
    weights_path = "checkpoints/best%s%s%s.weights" % ("2C" if opt.yu else "","BN" if opt.bn else "", "HR" if opt.hr else "")
    n_cpu = 4
    batch_size = 64
    channels = 2 if opt.yu else 3
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
        ListDataset(train_path,img_size=img_size, train=True, synth=finetune, yu=opt.yu), batch_size=batch_size, shuffle=True, num_workers=n_cpu
    )
    valloader = torch.utils.data.DataLoader(
        ListDataset(val_path,img_size=img_size, train=False, synth=finetune, yu=opt.yu), batch_size=batch_size, shuffle=False, num_workers=n_cpu
    )

    for transfer in transfers:
        if len(transfers) > 1:
            print("######################################################")
            print("############# Finetune with transfer: %d #############" % transfer)
            print("######################################################")
        for decay in decays:

            if len(decays) > 1:
                print("######################################################")
                print("############ Finetune with decay: %.1E ############" % decay)
                print("######################################################")

            torch.random.manual_seed(12345678)
            if cuda:
                torch.cuda.manual_seed(12345678)

            # Initiate model
            model = ROBO(inch=channels,bn=opt.bn,halfRes = halfRes)
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
            eta_min = learning_rate/25 if opt.transfer else learning_rate/10
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=eta_min)

            for epoch in range(epochs):
                #if finetune:
                train(epoch,epochs,100)
                bestLoss = valid(epoch,epochs,bestLoss,False)
                #else:
                    #bestLoss = train(epoch,epochs,bestLoss)

            if finetune and (transfer == 0):
                model.load_state_dict(torch.load("checkpoints/bestFinetune%s%s%s.weights" % ("2C" if opt.yu else "","BN" if opt.bn else "","HR" if opt.hr else "")))
                with torch.no_grad():
                    indices = pruneModel(model.parameters())

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate/40)
                print("Finetuning")

                bestLoss = 0

                for epoch in range(25):
                    train(epoch, 25, 100, indices=indices)
                    bestLoss = valid(epoch,25,bestLoss,True)
