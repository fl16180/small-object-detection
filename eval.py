import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer


def eval_mAP(data_loader, model, device):

    model.eval()

    # must compute mAP over entire dataset
    all_boxes = list()
    all_labels = list()
    all_scores = list()
    all_true_boxes = list()
    all_true_labels = list()
    all_difficulties = list()

    with torch.no_grad():
        for batch_idx, (data, boxes, labels, difficulties) in enumerate(data_loader):
            data = data.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            output_boxes, output_scores = model(data)

            batch_boxes, batch_labels, batch_scores = model.detect_objects(output_locs, output_scores,
                                                                           min_score=0.01, max_overlap=0.45,
                                                                           top_k=200)

            all_boxes.extend(batch_boxes)
            all_labels.extend(batch_labels)
            all_scores.extend(batch_scores)
            all_true_boxes.extend(boxes)
            all_true_labels.extend(labels)
            all_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(all_boxes, all_labels, all_scores, all_true_boxes, all_true_labels, all_difficulties)
