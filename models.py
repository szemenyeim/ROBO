from __future__ import division

import torch
import torch.nn as nn

from utils.utils import build_targets
from collections import defaultdict

class Conv(nn.Module):
    def __init__(self,inch,ch,stride=1,size=3,doBN = True):
        super(Conv,self).__init__()
        self.conv = nn.Conv2d(inch,ch,kernel_size=size,stride=stride,padding=size//2, bias=not doBN)
        self.bn = nn.BatchNorm2d(ch)
        self.relu = nn.LeakyReLU(0.1)

        self.size = size
        self.inch = inch
        self.stride = stride
        self.ch = ch
        self.doBN = doBN

    def forward(self, x):
        x = self.conv(x)
        if self.doBN:
            x = self.bn(x)
        return self.relu(x)

    def getComp(self,W,H):
        W = W // self.stride
        H = H // self.stride

        return self.size*self.size*W*H*self.inch*self.ch*2 + (W*H*self.ch*4 if self.doBN else 0), W, H

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 #+ num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(reduction='mean')  # Coordinate loss
        self.bce_loss = nn.BCELoss(reduction='mean')  # Confidence loss
        #self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, targets=None):
        nA = self.num_anchors
        nB = x.size(0)
        nGy = x.size(2)
        nGx = x.size(3)
        stride = self.image_dim / nGy

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = x.view(nB, nA, self.bbox_attrs, nGy, nGx).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        #pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nGx).repeat(nGy, 1).view([1, 1, nGy, nGx]).type(FloatTensor)
        grid_y = torch.arange(nGy).repeat(nGx, 1).t().view([1, 1, nGy, nGx]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.detach() + grid_x
        pred_boxes[..., 1] = y.detach() + grid_y
        pred_boxes[..., 2] = torch.exp(w.detach()) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.detach()) * anchor_h

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                #self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, corr = build_targets(
                pred_boxes=pred_boxes.cpu().detach(),
                pred_conf=pred_conf.cpu().detach(),
                #pred_cls=pred_cls.cpu().detach(),
                target=targets.cpu().detach(),
                anchors=scaled_anchors.cpu().detach(),
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size_y=nGy,
                grid_size_x=nGx,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            nCorrPrec = int((corr).sum().item())
            precision = float(nCorrPrec / nProposals) if nProposals > 0 else 0

            # Handle masks
            mask = mask.type(ByteTensor)
            conf_mask = conf_mask.type(ByteTensor)

            # Handle target variables
            tx = tx.type(FloatTensor)
            ty = ty.type(FloatTensor)
            tw = tw.type(FloatTensor)
            th = th.type(FloatTensor)
            tconf = tconf.type(FloatTensor)
            #tcls = tcls.type(LongTensor)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = 10*self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + 1*self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )
            #loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf #+ loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                0,
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    #pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            return output

class ROBO(nn.Module):
    def __init__(self, inch=3, ch=4, img_shape=(384,512), bn = False, halfRes=False):
        super(ROBO,self).__init__()

        self.img_shape = (img_shape[0] // 2,img_shape[1] // 2)  if halfRes else img_shape

        self.bn = bn
        self.halfRes = halfRes

        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

        self.branchLayers = [
            10 if halfRes else 11,
            -1
        ]

        self.anchors = [
            (42,39),
            (29,16),
            (31,109),
            (79,106),
        ]
        if bn:
            ch *= 2
            self.downPart = nn.ModuleList([
                Conv(inch,ch,2), # Stride: 2
                Conv(ch,ch*2,2), # Stride: 4
                Conv(ch*2,ch*4,2), # Stride: 8
                Conv(ch*4,ch*2,1,1),
                Conv(ch*2,ch*4,1),
                Conv(ch*4,ch*8,2), # Stride: 16
                Conv(ch*8,ch*4,1,1),
                Conv(ch*4,ch*8,1),
                Conv(ch*8,ch*16,2), # Stride: 32
                Conv(ch*16,ch*8,1,1),
                Conv(ch*8,ch*16,1),
                Conv(ch*16,ch*8,1,1),
                Conv(ch*8,ch*16,1), # First Classifier
                Conv(ch*16,ch*32,2), # Stride: 64
                Conv(ch*32,ch*16,1,1),
                Conv(ch*16,ch*32,1),
                Conv(ch*32,ch*16,1,1),
                Conv(ch*16,ch*32,1) # Second Classifier
            ])
            self.classifiers = nn.ModuleList([
                nn.Conv2d(ch*16,10,1),
                nn.Conv2d(ch*32,10,1)
            ])
        else:
            self.downPart = nn.ModuleList([
                None if halfRes else Conv(inch,ch,2), # Stride: 2
                Conv(inch if halfRes else ch,ch*2,2), # Stride: 4
                Conv(ch*2,ch*4,2), # Stride: 8
                Conv(ch*4,ch*4,1),
                Conv(ch*4,ch*8,2), # Stride: 16
                Conv(ch*8,ch*8,1),
                Conv(ch*8,ch*16,2), # Stride: 32
                Conv(ch*16,ch*16,1),
                Conv(ch*16,ch*16,1),
                Conv(ch*16,ch*16,1),
                Conv(ch*16,ch*16,1), # First Classifier
                Conv(ch*16,ch*32,2), # Stride: 64
                Conv(ch*32,ch*16,1),
                Conv(ch*16,ch*32,1),
                Conv(ch*32,ch*16,1),
                Conv(ch*16,ch*32,1) # Second Classifier
            ])
            self.classifiers = nn.ModuleList([
                nn.Conv2d(ch*16,10,1),
                nn.Conv2d(ch*32,10,1)
            ])
        self.yolo = nn.ModuleList([
            YOLOLayer(self.anchors[0:2], 2, img_shape[0]),
            YOLOLayer(self.anchors[2:4], 2, img_shape[0])
        ])

    def forward(self, x, targets = None):

        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        outNum = 0
        self.recprec = [0, 0, 0, 0]
        layer_outputs = [x]

        for layer in self.downPart:
            if layer is not None:
                layer_outputs.append(layer(layer_outputs[-1]))

        for idx, cl, yolo in zip(self.branchLayers,self.classifiers,self.yolo):
            out = cl(layer_outputs[idx])
            if is_training:
                out, *losses = yolo(out, targets[outNum])
                self.recprec[outNum * 2] += (losses[-2])
                self.recprec[outNum * 2 + 1] += (losses[-1])
                for name, loss in zip(self.loss_names, losses):
                    self.losses[name] += loss
            # Test phase: Get detections
            else:
                out = yolo(out)
            output.append(out)
            outNum += 1


        self.losses["recall"] /= outNum
        self.losses["precision"] /= outNum
        return sum(output) if is_training else torch.cat(output, 1)


    def get_computations(self,pruned = False):
        H, W = self.img_shape
        computations = []

        for module in self.downPart:
            if module is not None:
                ratio = float(module.conv.weight.nonzero().size(0)) / float(module.conv.weight.numel()) if pruned else 1
                if module is not None:
                    comp, W, H = module.getComp(W,H)
                    computations.append(comp * ratio)

        H, W = self.img_shape[0] // 32, self.img_shape[1] // 32
        computations.append(H*W*64*10*2 * (2 if self.bn else 1))
        computations.append(H*W*128*10//2 * (2 if self.bn else 1))

        return computations

