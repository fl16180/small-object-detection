from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True,
                    validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle,
                                                    validation_split, num_workers)


class VOCDataLoader(BaseDataLoader):
    """
    Load Pascal VOC using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0,
                        num_workers=1, training=True, mode='train'):
        # TODO: training vs mode
        trsfm = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
        # trsfm = None
        target_trsfm = None
        assert mode in ('train', 'trainval', 'val', 'test')
        self.data_dir = data_dir
        self.dataset = datasets.VOCDetection(self.data_dir, year='2012', image_set='train',
                                                    download=False, transform=trsfm,
                                                    target_transform=target_trsfm)
        super(VOCDataLoader, self).__init__(self.dataset, batch_size, shuffle,
                                                    validation_split, num_workers)
