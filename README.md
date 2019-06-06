# small-object-detection

CS231n project

Authors: Fred Lu, Samir Sen, Meera Srinivasan

## Overview
Experiments with different models for object detection on the Pascal VOC 2007 dataset.

See model/ directory for models: SSD300 and SSCoD. See <https://github.com/samirsen/small-object-detection/> for Faster R-CNN+GAN

The implementation of the novel spatial co-occurrence layer is in model/cooc_layers.py


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


## Acknowledgements
The following repos were essential to our work: 
<https://github.com/victoresque/pytorch-template>
<https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection>

The basic project backbone (loggers, model saving, base classes) was adapted from the first repo. I wrote the data loading and preprocessing by subclassing PyTorch modules. Some helper functions were closely adapted from the tutorial in the second link. It also guided me in writing the models and loss modules. The SSCoD model was written from scratch with help from useful PyTorch forum posts with utility functions.


