import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Pascal VOC constants
VOC_CLASS_NAMES = (
    '__background__',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
)
VOC_ENCODING = {cl: id for id, cl in enumerate(VOC_CLASS_NAMES)}
VOC_DECODING = {id: cl for id, cl in enumerate(VOC_CLASS_NAMES)}
VOC_NUM_CLASSES = 21

VOC_TRAIN_PARAMS = {
    "year": "2007",
    "image_set": "train"
}

VOC_VALID_PARAMS = {
    "year": "2007",
    "image_set": "trainval"
}

VOC_TEST_PARAMS = {
    "year": "2007",
    "image_set": "test"
}

# Next dataset constants
