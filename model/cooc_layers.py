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
    def __init__(self, in_channels, out_channels=32):
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

        x = F.relu(self.conv1(x))
        print(x.size())

        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        x = self.gaussian(x)

        print(x.size())
        N, C, H, W = x.size()

        # list of length H*W of (N, C, H, W) tensors containing each offset
        x_offsets = [self.roll(self.roll(x, i, 2), j, 3) for i in range(H) for j in range(W)]
        x_offsets = torch.cat(x_offsets, 1).to(DEVICE)   # (N, C*H*W, H, W)
        x_offsets = torch.view(N, C * H * W, H * W).permute(0, 2, 1)   # (N, H*W, C*H*W)

        x_base = x.view(N, C, H * W)    # (N, C, H*W)
        corrs = torch.bmm(x_base, x_offsets)    # (N, C, C*H*W)
        corrs = corrs.view(N, C * C, H * W)
        c_ij, best_offset = torch.max(corrs, 1)

        return c_ij

    @staticmethod
    def roll(tensor, shift, axis):
        """ https://discuss.pytorch.org/t/implementation-of-function-like-numpy-roll/964/6 """
        if shift == 0:
            return tensor

        if axis < 0:
            axis += tensor.dim()

        dim_size = tensor.size(axis)
        after_start = dim_size - shift
        if shift < 0:
            after_start = -shift
            shift = dim_size - abs(shift)

        before = tensor.narrow(axis, 0, dim_size - shift)
        after = tensor.narrow(axis, after_start, shift)
        return torch.cat([after, before], axis)


class SpatialCoocLayer(CoocLayer):
    """ Novel adaptation of co-occurrence layer for spatially localized
        co-occurrences. Instead of computing global co-occurrences over the
        entire activations, this returns a spatial activation map of local
        co-occurrences using pooling operations.
    """
    def __init__(self, in_channels, out_channels=32, local_kernel=5):
        super().__init__(in_channels, out_channels):

        self.avgpool = nn.AvgPool2d(kernel_size=local_kernel, stride=1)
        self.maxpool = nn.MaxPool2d()

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        x = self.gaussian(x)

        N, C, H, W = x.size()

        # list of length H*W of (N, C, H, W) tensors containing each offset
        x_offsets = [roll(roll(x, i, 2), j, 3) for i in range(H) for j in range(W)]
        x_offsets = torch.cat(x_offsets, 1).to(DEVICE)   # (N, C*H*W, H, W)

        x_base = x.repeat(1, H * W, 1, 1)       # (N, C*H*W, H, W)
        spatial_corrs = torch.mm(x, x_offsets)    # (N, C*H*W, H, W)

        self.avgpool(spatial_corrs)
        self.maxpool(spatial_corrs)

        # corrs = corrs.view(N, C * C, H * W)
        # c_ij, best_offset = torch.max(corrs, 1)

        return c_ij


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
