import torch
from torchvision import datasets, transforms
from base import BaseDataLoader


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


def parse_annotation_dict(annot):
    """ Parse the annotation dictionary for a single image/label set """

    objects = annot['annotation']['object']

    if isinstance(objects, list):
        labels = []
        boxes = []
        for o in objects:
            labels.append(VOC_ENCODING[o['name']])
            bbox = o['bndbox']
            boxes.append([int(bbox['xmin']), int(bbox['ymin']),
                          int(bbox['xmax']), int(bbox['ymax'])])

    elif isinstance(objects, dict):
        labels = [VOC_ENCODING[objects['name']]]
        bbox = objects['bndbox']
        boxes = [[int(bbox['xmin']), int(bbox['ymin']),
                      int(bbox['xmax']), int(bbox['ymax'])]]

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
    boxes = []
    labels = []

    for item in batch:
        images.append(item[0])
        box, label = parse_annotation_dict(item[1])
        boxes.append(box)
        labels.append(label)

    images = torch.stack(images, dim=0)

    return images, boxes, labels 


class VOCDataLoader(BaseDataLoader):
    """
    Load Pascal VOC using BaseDataLoader
    """
    def __init__(self, data_dir, image_size, batch_size,
                 collate_fn=collate_fn, shuffle=True, validation_split=0.0,
                 num_workers=1, training=True, augment=False, mode='train'):

        assert mode in ('train', 'trainval', 'val', 'test')

        if augment:
            image_trsfm = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.RandomCrop(200, 200),
                #transforms.Normalize((0.1307,), 0.3081,))
            ])
        else:
            image_trsfm = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        target_trsfm = None
        self.data_dir = data_dir
        self.dataset = datasets.VOCDetection(self.data_dir,
                                             year='2007',
                                             image_set='train',
                                             download=False,
                                             transform=image_trsfm,
                                             target_transform=target_trsfm)
        super(VOCDataLoader, self).__init__(self.dataset, batch_size,
                                            shuffle, validation_split,
                                            num_workers, collate_fn)
