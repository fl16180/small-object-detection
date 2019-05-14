import torch
from torchvision import datasets
from PIL import Image
import xml.etree.ElementTree as ET


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

        boxes, labels = parse_annotation_dict(target)

        if self.joint_transform is not None:
            img, boxes, labels = self.joint_transform(img, boxes, labels)

        return img, boxes, labels


def parse_annotation_dict(annot):
    """ Parse the annotation dictionary for a single image/label set. """

    objects = annot['annotation']['object']

    labels = []
    boxes = []
    if isinstance(objects, list):
        for o in objects:
            labels.append(VOC_ENCODING[o['name']])
            bbox = o['bndbox']
            boxes.append([int(bbox['xmin']) - 1, int(bbox['ymin']) - 1,
                          int(bbox['xmax']) - 1, int(bbox['ymax']) - 1])

    elif isinstance(objects, dict):
        labels.append(VOC_ENCODING[objects['name']])
        bbox = objects['bndbox']
        boxes.append([int(bbox['xmin']) - 1, int(bbox['ymin']) - 1,
                      int(bbox['xmax']) - 1, int(bbox['ymax']) - 1])

    boxes = torch.FloatTensor(boxes)
    labels = torch.LongTensor(labels)

    return boxes, labels
