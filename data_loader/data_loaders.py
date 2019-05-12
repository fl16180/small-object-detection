import sys
import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

from base import BaseDataLoader
from utils import SSDAugmentation

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


# Pascal VOC class to level
# TEMP: later code as dictionary from constants
VOC_ENCODING = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}


def im_box_transformer(image, boxes, labels):
    """ Custom transformer to jointly operate on an image and its
        corresponding bounding boxes and labels.

        The important steps are:
            Resize


        To perform data augmentation, additional steps are needed.
        (For now I will just wrap around the file in utils/joint_transforms.py)
    """
    # image = np.array(image)
    # trsfm = SSDAugmentation(size=300)
    # return trsfm(image, boxes, labels)

    tr = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor()])
    return tr(image), boxes, labels



def parse_annotation_dict(annot):
    """ Parse the annotation dictionary for a single image/label set """

    objects = annot['annotation']['object']

    labels = []
    boxes = []
    if isinstance(objects, list):
        for o in objects:
            labels.append(VOC_ENCODING[o['name']])
            bbox = o['bndbox']
            boxes.append([int(bbox['xmin']), int(bbox['ymin']),
                          int(bbox['xmax']), int(bbox['ymax'])])

    elif isinstance(objects, dict):
        labels.append(VOC_ENCODING[objects['name']])
        bbox = objects['bndbox']
        boxes.append([int(bbox['xmin']) - 1, int(bbox['ymin']) - 1,
                      int(bbox['xmax']) - 1, int(bbox['ymax']) - 1])

    boxes = torch.FloatTensor(boxes)
    labels = torch.LongTensor(labels)

    return boxes, labels


def collate_fn(batch):
    """ Collate objects together in a batch.

    The Pytorch dataset returns an image and annotation dictionary. We need to
    extract from the dictionary potentially multiple boxes and labels for each
    image.

    Inputs:
        batch: an iterable of N sets from __getitem__()

    Return:
        a tensor of images, list of varying-size tensors of bounding boxes,
        and list of vary-size tensors of encoded labels.
    """
    images = []
    boxes_list = []
    labels_list = []

    for item in batch:
        images.append(item[0])
        boxes_list.append(item[1])
        labels_list.append(item[2])

    images = torch.stack(images, dim=0)

    return images, boxes_list, labels_list


class ModVOCDetection(datasets.VOCDetection):

    def __init__(self, root, year='2012', image_set='train',
                 download=False, joint_transform=None):

        super(ModVOCDetection, self).__init__(root,
                                              year=year,
                                              image_set=image_set,
                                              download=download,
                                              transform=joint_transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of (boxes, labels)
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        boxes, labels = parse_annotation_dict(target)

        if self.transform is not None:
            img, boxes, labels = self.transform(img, boxes, labels)

        return img, boxes, labels


class VOCDataLoader(BaseDataLoader):
    """
    Load Pascal VOC using BaseDataLoader
    """
    def __init__(self, data_dir, image_size, batch_size,
                 collate_fn=collate_fn, shuffle=True, validation_split=0.0,
                 num_workers=1, training=True, augment=False, split='train'):

        assert split in ('train', 'trainval', 'val', 'test')

        # if augment:
        #     image_trsfm = transforms.Compose([
        #         transforms.Resize((image_size, image_size)),
        #         transforms.ToTensor(),
        #         transforms.RandomCrop(200, 200),
        #         #transforms.Normalize((0.1307,), 0.3081,))
        #     ])
        # else:
        #     image_trsfm = transforms.Compose([
        #         transforms.Resize((image_size, image_size)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])
        #     ])

        self.data_dir = data_dir
        self.dataset = ModVOCDetection(self.data_dir,
                                       year='2007',
                                       image_set='train',
                                       download=False,
                                       joint_transform=im_box_transformer)

        super(VOCDataLoader, self).__init__(self.dataset, batch_size,
                                            shuffle, validation_split,
                                            num_workers, collate_fn)
