import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import vgg16_bn
from base import BaseModel
from math import sqrt
from itertools import product as product
import torchvision

from utils import cxcy_to_gcxgcy, cxcy_to_xy, xy_to_cxcy, gcxgcy_to_cxcy
from utils import get_default_boxes, find_jaccard_overlap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Reference implementations of MultiBoxLoss at:

https://github.com/amdegroot/ssd.pytorch/blob/master/layers/modules/multibox_loss.py

https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/model.py

Computing it requires helper functions to generate prior boxes, compute
non-max suppression, and a few other things. Will need to (re-)implement these as well.

"""

def nll_loss(output, target):
    return F.nll_loss(output, target)

# ### VERSION 1
# class MultiBoxLoss(nn.Module):
#     """SSD Weighted Loss Function
#     Compute Targets:
#         1) Produce Confidence Target Indices by matching  ground truth boxes
#            with (default) 'priorboxes' that have jaccard index > threshold parameter
#            (default threshold: 0.5).
#         2) Produce localization target by 'encoding' variance into offsets of ground
#            truth boxes and their matched  'priorboxes'.
#         3) Hard negative mining to filter the excessive number of negative examples
#            that comes with using a large number of default bounding boxes.
#            (default negative:positive ratio 3:1)
#     Objective Loss:
#         L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
#         Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
#         weighted by α which is set to 1 by cross val.
#         Args:
#             c: class confidences,
#             l: predicted boxes,
#             g: ground truth boxes
#             N: number of matched default boxes
#         See: https://arxiv.org/pdf/1512.02325.pdf for more details.
#     """
#
#     def __init__(self, num_classes, overlap_thresh, prior_for_matching,
#                  bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
#                  use_gpu=True):
#         super(MultiBoxLoss, self).__init__()
#         self.use_gpu = use_gpu
#         self.num_classes = num_classes
#         self.threshold = overlap_thresh
#         self.background_label = bkg_label
#         self.encode_target = encode_target
#         self.use_prior_for_matching = prior_for_matching
#         self.do_neg_mining = neg_mining
#         self.negpos_ratio = neg_pos
#         self.neg_overlap = neg_overlap
#         self.variance = cfg['variance']
#
#     def forward(self, predictions, targets):
#         """Multibox Loss
#         Args:
#             predictions (tuple): A tuple containing loc preds, conf preds,
#             and prior boxes from SSD net.
#                 conf shape: torch.size(batch_size,num_priors,num_classes)
#                 loc shape: torch.size(batch_size,num_priors,4)
#                 priors shape: torch.size(num_priors,4)
#             targets (tensor): Ground truth boxes and labels for a batch,
#                 shape: [batch_size,num_objs,5] (last idx is the label).
#         """
#         loc_data, conf_data, priors = predictions
#         num = loc_data.size(0)
#         priors = priors[:loc_data.size(1), :]
#         num_priors = (priors.size(0))
#         num_classes = self.num_classes
#
#         # match priors (default boxes) and ground truth boxes
#         loc_t = torch.Tensor(num, num_priors, 4)
#         conf_t = torch.LongTensor(num, num_priors)
#         for idx in range(num):
#             truths = targets[idx][:, :-1].data
#             labels = targets[idx][:, -1].data
#             defaults = priors.data
#             match(self.threshold, truths, defaults, self.variance, labels,
#                   loc_t, conf_t, idx)
#         if self.use_gpu:
#             loc_t = loc_t.cuda()
#             conf_t = conf_t.cuda()
#         # wrap targets
#         loc_t = Variable(loc_t, requires_grad=False)
#         conf_t = Variable(conf_t, requires_grad=False)
#
#         pos = conf_t > 0
#         num_pos = pos.sum(dim=1, keepdim=True)
#
#         # Localization Loss (Smooth L1)
#         # Shape: [batch,num_priors,4]
#         pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
#         loc_p = loc_data[pos_idx].view(-1, 4)
#         loc_t = loc_t[pos_idx].view(-1, 4)
#         loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
#
#         # Compute max conf across batch for hard negative mining
#         batch_conf = conf_data.view(-1, self.num_classes)
#         loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
#
#         # Hard Negative Mining
#         loss_c[pos] = 0  # filter out pos boxes for now
#         loss_c = loss_c.view(num, -1)
#         _, loss_idx = loss_c.sort(1, descending=True)
#         _, idx_rank = loss_idx.sort(1)
#         num_pos = pos.long().sum(1, keepdim=True)
#         num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
#         neg = idx_rank < num_neg.expand_as(idx_rank)
#
#         # Confidence Loss Including Positive and Negative Examples
#         pos_idx = pos.unsqueeze(2).expand_as(conf_data)
#         neg_idx = neg.unsqueeze(2).expand_as(conf_data)
#         conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
#         targets_weighted = conf_t[(pos+neg).gt(0)]
#         loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
#
#         # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
#
#         N = num_pos.data.sum()
#         loss_l /= N
#         loss_c /= N
#         return loss_l, loss_c


### VERSION 2
class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = get_default_boxes()
        self.priors_xy = cxcy_to_xy(self.priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss
