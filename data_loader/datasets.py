import torch
from torchvision import datasets
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

from constants import VOC_ENCODING


class ModVOCDetection(datasets.VOCDetection):
    """ Inherits PyTorch implementation of VOCDetection Dataset with
        necessary modifications to support joint transformations.
    """
    def __init__(self, root, year='2007', image_set='train',
                 download=False, joint_transform=None):

        self.joint_transform = joint_transform
        super(ModVOCDetection, self).__init__(root,
                                              year=year,
                                              image_set=image_set,
                                              download=download)

    def __getitem__(self, index):
        """ Return a tuple consisting of the image, its bounding boxes,
            and the labels of the bounding boxes.

        Args:
            index (int): Index
        Returns:
            tuple: (image, boxes, labels)
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        boxes, labels, difficulties = parse_annotation_dict(target)

        if self.joint_transform is not None:
            img, boxes, labels = self.joint_transform(img, boxes, labels)

        return img, boxes, labels, torch.LongTensor(difficulties)


def parse_annotation_dict(annot):
    """ Parse the annotation dictionary for a single image/label set.

        Annotations are stored in a nested dictionary. We require three
        elements:
            1. labels of any bounding boxes
            2. corner coordinates of any bounding boxes
            3. "difficulty" of the detected object, used later for metrics
    """

    objects = annot['annotation']['object']

    labels = []
    boxes = []
    difficulties = []
    if isinstance(objects, list):
        for o in objects:
            labels.append(VOC_ENCODING[o['name']])
            bbox = o['bndbox']
            boxes.append([int(bbox['xmin']) - 1, int(bbox['ymin']) - 1,
                          int(bbox['xmax']) - 1, int(bbox['ymax']) - 1])
            difficulties.append(o['difficult'])

    elif isinstance(objects, dict):
        labels.append(VOC_ENCODING[objects['name']])
        bbox = objects['bndbox']
        boxes.append([int(bbox['xmin']) - 1, int(bbox['ymin']) - 1,
                      int(bbox['xmax']) - 1, int(bbox['ymax']) - 1])
        difficulties.append(objects['difficult'])

    boxes = np.array(boxes, dtype=np.float32)
    labels = np.array(labels)
    difficulties = np.array(difficulties).astype(int)

    return boxes, labels, difficulties
