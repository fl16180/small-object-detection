# small-object-detection
CS231n project
Authors: Fred Lu, Samir Sen, Meera Srinivasan

## Overview
Experiments with different models for object detection on the Pascal VOC 2007 dataset.

See model/ directory for models: SSD300, SSCoD, Faster R-CNN+GAN.

Our implementation of the novel spatial co-occurrence layer is in model/cooc_layers.py


### Requirements
PyTorch >=1.0 with torchvision, image processing libraries (PIL, cv2). Python 3.7 Anaconda distribution.

## Instructions
Set hyperparameters in config.json.

Run 
``` 
python train.py -c config.json
```

Model checkpoints are automatically saved. Resume training with 
```
python train.py -r saved/models/path-to-checkpoint.pth
```

Remaining scripts in root directory are self-explanatory, e.g. evaluation and producing images with bounding boxes.

