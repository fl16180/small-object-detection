import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

from base import BaseModel
from utils import *

from math import sqrt
from itertools import product as product


class VGG16(nn.Module):
    """ Modified VGG16 base for generating features from image
    """
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Replace the VGG16 FC layers with additional conv2d (see Fig. 2)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        # Load pretrained layers
        self.load_pretrained_layers()

    def forward(self, x):

        out = F.relu(self.conv1_1(x))  # (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = F.relu(self.conv2_1(out))  # (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = F.relu(self.conv3_1(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_2(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38) (note ceil_mode=True)

        out = F.relu(self.conv4_1(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_2(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out))  # (N, 512, 38, 38)
        conv4_out = out  # (N, 512, 38, 38)
        out = self.pool4(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv5_1(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out))  # (N, 512, 19, 19)
        out = self.pool5(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv6(out))  # (N, 1024, 19, 19)

        conv7_out = F.relu(self.conv7(out))  # (N, 1024, 19, 19)

        return conv4_out, conv7_out

    def load_pretrained_layers(self):
        """
        (This function as defined in https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection))

        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
        # fc6
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

        # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
        # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
        # ...operating on the 2D image of size (C, H, W) without padding

        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")


class ExtraLayers(nn.Module):
    """ Additional convolutions after VGG16 for feature scaling. """
    def __init__(self):
        super(ExtraLayers, self).__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.)

    def forward(self, conv7_out):

        out = F.relu(self.conv8_1(conv7_out))  # (N, 256, 19, 19)
        out = F.relu(self.conv8_2(out))  # (N, 512, 10, 10)
        conv8_out = out  # (N, 512, 10, 10)

        out = F.relu(self.conv9_1(out))  # (N, 128, 10, 10)
        out = F.relu(self.conv9_2(out))  # (N, 256, 5, 5)
        conv9_out = out  # (N, 256, 5, 5)

        out = F.relu(self.conv10_1(out))  # (N, 128, 5, 5)
        out = F.relu(self.conv10_2(out))  # (N, 256, 3, 3)
        conv10_out = out  # (N, 256, 3, 3)

        out = F.relu(self.conv11_1(out))  # (N, 128, 3, 3)
        conv11_out = F.relu(self.conv11_2(out))  # (N, 256, 1, 1)

        return conv8_out, conv9_out, conv10_out, conv11_out


class Classifiers(nn.Module):

    def __init__(self, n_classes, n_boxes):
        super(Classifiers, self).__init__()

        self.n_classes = n_classes
        self.n_boxes = n_boxes
        assert len(self.n_boxes) == 6

        self.box4 = nn.Conv2d(512, n_boxes[0] * 4, kernel_size=3, padding=1)
        self.class4 = nn.Conv2d(512, n_boxes[0] * n_classes, kernel_size=3, padding=1)

        self.box7 = nn.Conv2d(1024, n_boxes[1] * 4, kernel_size=3, padding=1)
        self.class7 = nn.Conv2d(1024, n_boxes[1] * n_classes, kernel_size=3, padding=1)

        self.box8 = nn.Conv2d(512, n_boxes[2] * 4, kernel_size=3, padding=1)
        self.class8 = nn.Conv2d(512, n_boxes[2] * n_classes, kernel_size=3, padding=1)

        self.box9 = nn.Conv2d(256, n_boxes[3] * 4, kernel_size=3, padding=1)
        self.class9 = nn.Conv2d(256, n_boxes[3] * n_classes, kernel_size=3, padding=1)

        self.box10 = nn.Conv2d(256, n_boxes[4] * 4, kernel_size=3, padding=1)
        self.class10 = nn.Conv2d(256, n_boxes[4] * n_classes, kernel_size=3, padding=1)

        self.box11 = nn.Conv2d(256, n_boxes[5] * 4, kernel_size=3, padding=1)
        self.class11 = nn.Conv2d(256, n_boxes[5] * n_classes, kernel_size=3, padding=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.)

    def forward(self, inputs):
        conv4_out, conv7_out, conv8_out, conv9_out, conv10_out, conv11_out = inputs

        N = conv4_out.size(0)

        # swap dimensions around and reassign in natural layout (contiguous)
        box4 = self.box4(conv4_out).permute(0, 2, 3, 1).contiguous()
        box7 = self.box7(conv7_out).permute(0, 2, 3, 1).contiguous()
        box8 = self.box8(conv8_out).permute(0, 2, 3, 1).contiguous()
        box9 = self.box9(conv9_out).permute(0, 2, 3, 1).contiguous()
        box10 = self.box10(conv10_out).permute(0, 2, 3, 1).contiguous()
        box11 = self.box11(conv11_out).permute(0, 2, 3, 1).contiguous()

        class4 = self.class4(conv4_out).permute(0, 2, 3, 1).contiguous()
        class7 = self.class7(conv7_out).permute(0, 2, 3, 1).contiguous()
        class8 = self.class8(conv8_out).permute(0, 2, 3, 1).contiguous()
        class9 = self.class9(conv9_out).permute(0, 2, 3, 1).contiguous()
        class10 = self.class10(conv10_out).permute(0, 2, 3, 1).contiguous()
        class11 = self.class11(conv11_out).permute(0, 2, 3, 1).contiguous()

        # reshape to match expected bounding box and class score shapes
        box4 = box4.view(N, -1, 4)
        box7 = box7.view(N, -1, 4)
        box8 = box8.view(N, -1, 4)
        box9 = box9.view(N, -1, 4)
        box10 = box10.view(N, -1, 4)
        box11 = box11.view(N, -1, 4)

        class4 = class4.view(N, -1, self.n_classes)
        class7 = class7.view(N, -1, self.n_classes)
        class8 = class8.view(N, -1, self.n_classes)
        class9 = class9.view(N, -1, self.n_classes)
        class10 = class10.view(N, -1, self.n_classes)
        class11 = class11.view(N, -1, self.n_classes)

        boxes = torch.cat([box4, box7, box8, box9, box10, box11], dim=1)
        classes = torch.cat([class4, class7, class8, class9, class10, class11],
                            dim=1)

        return boxes, classes


class SSD300(nn.Module):
    """ The full SSD300 network.

        The full process involves:
            1) run VGG16 base on the image and extract layer 4 & 7 features.
            2) run ExtraLayers to systematically downscale image while pulling
                features from each scaling.
            3) run Classifiers to compute boxes and classes for each
                feature set.
    """

    def __init__(self, n_classes, n_boxes=(4, 6, 6, 6, 4, 4)):
        super(SSD300, self).__init__()

        self.n_classes = n_classes
        self.n_boxes = n_boxes

        self.base = VGG16()
        self.extra = ExtraLayers()
        self.classifiers = Classifiers(n_classes, n_boxes)

        # L2 norm scaler for conv4_out. Updated thru backprop
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)

        # default boxes
        self.priors = get_default_boxes()

    def forward(self, image):
        """ Forward propagation.

            Input: images forming tensor of dimensions (N, 3, 300, 300)

            Returns: 8732 locations and class scores for each image.
        """
        # Run VGG16
        conv4_out, conv7_out = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)
        conv4_out = self.L2Norm(conv4_out)

        # Run ExtraLayers
        conv8_out, conv9_out, conv10_out, conv11_out = self.extra(conv7_out)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # setup prediction inputs
        features = (conv4_out, conv7_out, conv8_out, conv9_out,
                    conv10_out, conv11_out)

        # Run Classifiers
        output_boxes, output_scores = self.classifiers(features)

        return output_boxes, output_scores

    def L2Norm(self, out, eps=1e-10):
        """ Rescale the outputs of conv4. The rescaling factor is a parameter
            that gets updated through backprop.
        """
        norm = out.pow(2).sum(dim=1, keepdim=True).sqrt() + eps
        out = torch.div(out, norm)
        return out * self.rescale_factors

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        (hasn't been rewritten yet)

        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]))
                image_labels.append(torch.LongTensor([0]))
                image_scores.append(torch.FloatTensor([0.]))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size
