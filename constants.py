

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
VOC_ENCODING = {class: id for id, class in enumerate(VOC_CLASS_NAMES)}
VOC_DECODING = {id: class for id, class in enumerate(VOC_CLASS_NAMES)}

VOC_NUM_CLASSES = 20
