from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import sys

import argparse
import progressbar

import torch
from torch.utils.data import DataLoader

def getmAP(all_annotations,all_detections):
    mAPs = np.zeros((2, 5))
    APs = np.zeros((2, 4, 5))
    thresholds = np.array([[4, 8, 16, 32, 64], [0.75, 0.5, 0.25, 0.1, 0.05]])
    for useIoU in range(2):
        for threshIdx in range(5):
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

                        if useIoU > 0:
                            overlaps = bbox_iou_numpy(np.expand_dims(bbox, axis=0), annotations)
                            assigned_annotation = np.argmax(overlaps, axis=1)
                            max_overlap = overlaps[0, assigned_annotation]

                            if max_overlap >= thresholds[
                                useIoU, threshIdx] and assigned_annotation not in detected_annotations:
                                true_positives.append(1)
                                detected_annotations.append(assigned_annotation)
                            else:
                                true_positives.append(0)
                        else:
                            distances = bbox_dist(bbox, annotations)
                            assigned_annotation = np.argmin(distances)
                            min_dist = distances[assigned_annotation]

                            if min_dist <= thresholds[
                                useIoU, threshIdx] and assigned_annotation not in detected_annotations:
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

            for c, ap in average_precisions.items():
                APs[useIoU, c, threshIdx] = ap

            mAP = np.mean(list(average_precisions.values()))
            mAPs[useIoU, threshIdx] = mAP
    return  mAPs, APs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default="config/robo-down-small.cfg", help="path to model config file")
    parser.add_argument("--class_path", type=str, default="data/robo.names", help="path to class label file")
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.45, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=(384,512), help="size of each image dimension")
    parser.add_argument("--transfer", help="Layers to truly train", action="store_true", default=False)
    parser.add_argument("--finetune", help="Finetuning", action="store_true", default=False)
    parser.add_argument("--bn", help="Use bottleneck", action="store_true", default=False)
    parser.add_argument("--yu", help="Use 2 channels", action="store_true", default=False)
    parser.add_argument("--hr", help="Use half res", action="store_true", default=False)
    parser.add_argument("--lprop", help="Use half res", action="store_true", default=False)
    opt = parser.parse_args()

    cuda = torch.cuda.is_available()

    lprop = opt.lprop

    data_config_path = "config/roboFinetune.data" if opt.finetune else "config/robo.data"
    img_size = (192,256) if opt.hr else (384,512)

    name = "checkpoints/bestFinetune" if opt.finetune else "checkpoints/best"
    if opt.yu:
        name += "2C"
    if opt.bn:
        name += "BN"
    if opt.hr:
        name += "HR"

    weights_path = []
    if opt.transfer:
        weights_path = sorted(glob.glob(name + "T*.weights"),reverse=True)
    elif opt.finetune:
        weights_path = sorted(glob.glob(name + "*_*.weights"),reverse=True)
    weights_path += [name + ".weights"]
    if not opt.bn:
        weights_path = [path for path in weights_path if "BN" not in path]
    if not opt.yu:
        weights_path = [path for path in weights_path if "2C" not in path]
    if not opt.hr:
        weights_path = [path for path in weights_path if "HR" not in path]
    if lprop:
        weights_path = [weights_path[0]]

    # Get data configuration
    data_config = parse_data_config(data_config_path)
    test_path = data_config["valid"]
    if lprop:
        test_path = "../../Data/RoboCup" if sys.platform != 'win32' else "D:/Datasets/RoboCup"
    num_classes = int(data_config["classes"])
    channels = 2 if opt.yu else 3
    seq_len = 4

    # Initiate model
    for path in weights_path:
        print(path)
        model = ROBO(inch=channels,bn=opt.bn, halfRes=opt.hr)
        print(model.getParams())
        exit(0)
        model.load_state_dict(torch.load(path,map_location={'cuda:0': 'cpu'}))

        print(count_zero_weights(model))

        #with torch.no_grad():
            #pruneModel(model.parameters())

        computations = model.get_computations(True)

        print(computations)
        print(sum(computations))

        if cuda:
            model = model.cuda()

        model.eval()

        # Get dataloader
        if lprop:
            dataset = LPDataSet(test_path, train=False, finetune=opt.finetune, yu=opt.yu, img_size=img_size,len_seq=seq_len)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.n_cpu, collate_fn=my_collate)
        else:
            dataset = ListDataset(test_path, train=False, synth=opt.finetune, yu=opt.yu, img_size=img_size)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        print("Compute mAP...")

        all_detections = []
        all_lp_detections = []
        all_annotations = []

        bar = progressbar.ProgressBar(0, len(dataloader), redirect_stdout=False)

        for batch_i, data in enumerate(dataloader):

            if lprop:
                imgs, targets, cvimgs = data
                cvimgs = cvimgs[0]
            else:
                _, imgs, targets = data
            imgs = imgs.type(Tensor)

            with torch.no_grad():
                outputs = model(imgs)
                outputs = non_max_suppression(outputs, 80, conf_thres=opt.conf_thres, nms_thres=opt.nms_thres)

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
                    annotation_boxes[:, 0] = (_annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2)*opt.img_size[1]
                    annotation_boxes[:, 1] = (_annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2)*opt.img_size[0]
                    annotation_boxes[:, 2] = (_annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2)*opt.img_size[1]
                    annotation_boxes[:, 3] = (_annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2)*opt.img_size[0]
                    #annotation_boxes *= opt.img_size

                    for label in range(num_classes):
                        all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

            if lprop:
                for i in range(seq_len):
                    if i == 0:
                        all_lp_detections.append(labelProp(cvimgs[i],cvimgs[i+1],all_detections[-seq_len+1]))
                    else:
                        all_lp_detections.append(labelProp(cvimgs[i],cvimgs[i-1],all_detections[-seq_len+i-1]))

            bar.update(batch_i)
        bar.finish()

        mAPs, APs = getmAP(all_annotations,all_detections)
        if not lprop:
            for c in range(4):
                print("Class %d:" % c)
                for i in range(2):
                    print("Dist: " if i < 1 else "IoU: ",APs[i,c,:])
            print("mAP:")
        for i in range(2):
            print("Dist: " if i < 1 else "IoU: ",mAPs[i,:])

        if lprop:
            LPmAPs, LPAPs = getmAP(all_annotations, all_lp_detections)
            print("LP")
            print("mAP:")
            for i in range(2):
                print("Dist: " if i < 1 else "IoU: ",LPmAPs[i,:])
