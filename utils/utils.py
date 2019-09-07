from __future__ import division
import math
import time
import torch
import torch.nn as nn
import numpy as np
import glob
from PIL import Image
import progressbar
import cv2
import os


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_iou_numpy(box1, box2):
    """Computes IoU between bounding boxes.
    Parameters
    ----------
    box1 : ndarray
        (N, 4) shaped array with bboxes
    box2 : ndarray
        (M, 4) shaped array with bboxes
    Returns
    -------
    : ndarray
        (N, M) shaped array with IoUs
    """
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        classPred = torch.cat((torch.zeros(192),torch.ones(192),2*torch.ones(48),3*torch.ones(48))).unsqueeze(1)
        if torch.cuda.is_available():
            classPred = classPred.cuda()
        classPred = classPred[conf_mask]
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf = image_pred[:, 4].unsqueeze(1)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), classPred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def labelProp(img_gr,prevImg,BBS):
    of = cv2.calcOpticalFlowFarneback(prevImg, img_gr, None, pyr_scale=0.5,levels=2,winsize=15,iterations=2,poly_n=7,poly_sigma=1.5,flags=0)
    scale = 4.0
    ret = []
    for classBB in BBS:
        newClassBB = []
        for BB in classBB:
            xMin = max(0, int(BB[0]/scale))
            yMin = max(0, int(BB[1]/scale))
            xMax = min(img_gr.shape[1] - 1, int(math.ceil(BB[2]/scale)))
            yMax = min(img_gr.shape[0] - 1, int(math.ceil(BB[3]/scale)))
            patch = of[yMin:yMax, xMin:xMax]
            meanX = np.mean(patch[:, :, 0])*scale
            meanY = np.mean(patch[:, :, 1])*scale
            newBB = []
            newBB.append(max(0, int(round(BB[0] + meanX))))
            newBB.append(max(0, int(round(BB[1] + meanY))))
            newBB.append(min(img_gr.shape[1]*scale - 1, int(round(BB[2] + meanX))))
            newBB.append(min(img_gr.shape[0]*scale - 1, int(round(BB[3] + meanY))))
            newBB.append(BB[4])
            newClassBB.append(newBB)
        ret.append(newClassBB)
    return ret

def pruneModel(params, ratio = 0.01, glasso=False):
    i = 0
    indices = []
    for param in params:
        if param.dim() > 1:
            if glasso:
                dim = param.size()
                if dim.__len__() > 2:
                    ind = torch.zeros_like(param)
                    filtCnt = 0
                    vals = param.pow(2).sum(dim=(1,2,3)).add(1e-8).pow(1 / 2.)
                    thresh = torch.max(vals) * ratio
                    for i,v in enumerate(vals):
                        if v < thresh:
                            filtCnt += 1
                            param[i,:] = torch.zeros_like(param[i])
                            ind[i,:] = torch.ones_like(ind[i])
                    print("Pruned %f%% of the filters" % (filtCnt/vals.numel()*100))
                    indices.append(ind.bool())
                else:
                    indices.append(torch.zeros_like(param).bool())
            else:
                thresh = torch.max(torch.abs(param)) * ratio
                print("Pruned %f%% of the weights" % (
                float(torch.sum(torch.abs(param) < thresh)) / float(torch.sum(param != 0)) * 100))
                param[torch.abs(param) < thresh] = 0
                indices.append(torch.abs(param) < thresh)
            i += 1

    return indices

def count_zero_weights(model,glasso=False):
    nonzeroWeights = 0
    totalWeights = 0
    if glasso:
        for param in model.parameters():
            dim = param.size()
            if dim.__len__() > 2:
                vals = param.pow(2).sum(dim=(1,2,3)).add(1e-8).pow(1/2.)
                max = torch.max(vals)
                nonzeroWeights += (vals < max * 0.01).sum().float()
                totalWeights += vals.numel()
    else:
        for param in model.parameters():
            max = torch.max(torch.abs(param))
            nonzeroWeights += (torch.abs(param) < max*0.01).sum().float()
            totalWeights += param.numel()
    return float(nonzeroWeights/totalWeights)

def build_targets(
    pred_boxes, pred_conf, target, anchors, num_anchors, num_classes, grid_size_y, grid_size_x, ignore_thres, img_dim
):
    nB = target.size(0)
    nA = num_anchors
    #nC = num_classes
    nGx = grid_size_x
    nGy = grid_size_y
    mask = torch.zeros(nB, nA, nGy, nGx)
    conf_mask = torch.ones(nB, nA, nGy, nGx)
    tx = torch.zeros(nB, nA, nGy, nGx)
    ty = torch.zeros(nB, nA, nGy, nGx)
    tw = torch.zeros(nB, nA, nGy, nGx)
    th = torch.zeros(nB, nA, nGy, nGx)
    tconf = torch.ByteTensor(nB, nA, nGy, nGx).fill_(0)
    corr = torch.ByteTensor(nB, nA, nGy, nGx).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            # Convert to position relative to box
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            gx = target[b, t, 1] * nGx
            gy = target[b, t, 2] * nGy
            gw = target[b, t, 3] * nGx
            gh = target[b, t, 4] * nGy
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)

            best_n = target_label
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            #tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            score = pred_conf[b, best_n, gj, gi]
            if (target_label != 3 or iou > 0.5) and score > 0.5:
                nCorrect += 1
                corr[b, best_n, gj, gi] = 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, corr


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype="uint8")[y])

def bbox_dist(box1,boxes):
    distances = np.array([])
    for box2 in boxes:
        cent1x = (box1[0] + box1[2]) / 2
        cent1y = (box1[1] + box1[3]) / 2
        cent2x = (box2[0] + box2[2]) / 2
        cent2y = (box2[1] + box2[3]) / 2
        distances = np.append(distances,np.sqrt(pow(cent1x-cent2x,2) + pow(cent1y-cent2y,2)))
    return distances

def computeAP(model,dataloader,conf_thres,nms_thres,num_classes,img_size,useIoU,thresh):
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    all_detections = []
    all_annotations = []

    bar = progressbar.ProgressBar(0, len(dataloader), redirect_stdout=False)

    for batch_i, (_, imgs, targets) in enumerate(dataloader):

        if torch.cuda.is_available():
            imgs = imgs.cuda()

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, 80, conf_thres=conf_thres, nms_thres=nms_thres)

        for output, annotations in zip(outputs, targets):

            all_detections.append([np.array([]) for _ in range(num_classes)])
            if output is not None:
                # Get predicted boxes, confidence scores and labels
                pred_boxes = output[:, :5].cpu().numpy()
                scores = output[:, 4].cpu().numpy()
                pred_labels = output[:, -1].cpu().numpy()

                # Order by confidence
                sort_i = np.argsort(scores)
                pred_labels = pred_labels[sort_i]
                pred_boxes = pred_boxes[sort_i]

                for label in range(num_classes):
                    all_detections[-1][label] = pred_boxes[pred_labels == label]

            all_annotations.append([np.array([]) for _ in range(num_classes)])
            if any(annotations[:, -1] > 0):

                annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
                _annotation_boxes = annotations[annotations[:, -1] > 0, 1:]

                # Reformat to x1, y1, x2, y2 and rescale to image dimensions
                annotation_boxes = np.empty_like(_annotation_boxes)
                annotation_boxes[:, 0] = (_annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2) * img_size[1]
                annotation_boxes[:, 1] = (_annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2) * img_size[0]
                annotation_boxes[:, 2] = (_annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2) * img_size[1]
                annotation_boxes[:, 3] = (_annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2) * img_size[0]
                # annotation_boxes *= opt.img_size

                for label in range(num_classes):
                    all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

        bar.update(batch_i)
    bar.finish()
    average_precisions = {}
    for label in range(num_classes):
        true_positives = []
        scores = []
        num_annotations = 0

        for i in range(len(all_annotations)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]

            num_annotations += annotations.shape[0]
            detected_annotations = []

            for *bbox, score in detections:
                scores.append(score)

                if annotations.shape[0] == 0:
                    true_positives.append(0)
                    continue

                if useIoU:
                    overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= thresh and assigned_annotation not in detected_annotations:
                        true_positives.append(1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        true_positives.append(0)
                else:
                    distances = bbox_dist(bbox, annotations)
                    assigned_annotation = np.argmin(distances)
                    min_dist = distances[assigned_annotation]

                    if min_dist <= thresh and assigned_annotation not in detected_annotations:
                        true_positives.append(1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        true_positives.append(0)

        # no annotations -> AP for this class is 0
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        true_positives = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives
        # sort by score
        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision

    mAP = np.mean(list(average_precisions.values()))

    return mAP,list(average_precisions.values())
