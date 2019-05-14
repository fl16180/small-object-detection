

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

VOC_NUM_CLASSES = 20
