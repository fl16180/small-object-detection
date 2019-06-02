import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from itertools import product
from scipy.ndimage.filters import gaussian_filter

from base import BaseModel
from utils import *
from constants import *



class CoocLayer(nn.Module):
    """ Co-occurrence layer as proposed in Shih et al. (CVPR 2017)
    """
    def __init__(self, in_channels, out_channels=32, offset=5):
        super().__init__()

        # dimensionality reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # gaussian filtering for noise removal
        self.gaussian = GaussianSmoothing(channels=out_channels, kernel_size=5, sigma=1)

    def forward(self, x):
        """ input is the image activations following a convolutional layer.
            dimensions: N x C x H x W

            The co-occurrence layer computes a vector of length C ** 2
        """

        N, C, H, W = x.size()
        print(x.size())
        x = F.relu(self.conv1(x))

        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        x = self.gaussian(x)
        print(x.size())
        for pair in product(range(C), repeat=2):
            Ai = x[N, pair[0], :, :]
            Aj = x[N, pair[1], :, :]





        out = F.relu(self.conv1_1(x))  # (N, 64, 300, 300)
        out = F.relu(self.conv1_2(out))  # (N, 64, 300, 300)
        out = self.pool1(out)  # (N, 64, 150, 150)

        out = F.relu(self.conv2_1(out))  # (N, 128, 150, 150)
        out = F.relu(self.conv2_2(out))  # (N, 128, 150, 150)
        out = self.pool2(out)  # (N, 128, 75, 75)

        out = F.relu(self.conv3_1(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_2(out))  # (N, 256, 75, 75)
        out = F.relu(self.conv3_3(out))  # (N, 256, 75, 75)
        out = self.pool3(out)  # (N, 256, 38, 38) (note ceil_mode=True)

        out = F.relu(self.conv4_1(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_2(out))  # (N, 512, 38, 38)
        out = F.relu(self.conv4_3(out))  # (N, 512, 38, 38)
        conv4_out = out  # (N, 512, 38, 38)
        out = self.pool4(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv5_1(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_2(out))  # (N, 512, 19, 19)
        out = F.relu(self.conv5_3(out))  # (N, 512, 19, 19)
        out = self.pool5(out)  # (N, 512, 19, 19)

        out = F.relu(self.conv6(out))  # (N, 1024, 19, 19)

        conv7_out = F.relu(self.conv7(out))  # (N, 1024, 19, 19)

        return conv4_out, conv7_out

class SpatialCoocLayer(nn.Module):
    def __init__(self):
        super().__init__():

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.gaussian_filter = nn.Conv2d(out_channels, out_channels,
                                         kernel_size=3, padding=1,
                                         groups=out_channels, bias=False)
        with torch.no_grad():
            self.gaussian_filter.weight = gaussian_weights


        self.avgpool = nn.AvgPool2d()
        self.maxpool = nn.MaxPool2d()

    def forward(self, ):

        for pair in product(range(C), repeat=2):
            Ai = x[N, pair[0], :, :]
            Aj = x[N, pair[1], :, :]
            Ai * Aj

class GaussianSmoothing(nn.Module):
    """
    (From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10)
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
