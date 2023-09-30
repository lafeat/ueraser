import io
import math
import random
from typing import Optional, Protocol

import torch
from torch import nn, Tensor
import einops
import numpy as np
from PIL import Image
import kornia.augmentation as K


def jpeg_compression(
    in_tensor: Tensor, quality: Optional[int] = None
) -> Tensor:
    """
    This function performs JPEG compression of a batch of images.

    in_tensor:
        A batch of images in the range [0, 1] and shape (B, C, H, W),
        where B is the batch size, C is the number of channels,
        H is the height, and W is the width.
    quality:
        The JPEG quality factor. If None, a random quality factor is chosen
        uniformly from the range [10, 100].

    Returns:
        A batch of JPEG-compressed images in the range [0, 1]
        with shape (B, C, H, W).
    """
    batch_size = in_tensor.size(0)
    # optionally choose a random quality factor
    quality = quality or random.randrange(10, 100)
    # pad the batch size to a square number
    bsqr = int(math.ceil(math.sqrt(batch_size)))
    bpad = bsqr ** 2 - batch_size
    if bpad:
        zeros = torch.zeros(
            bpad, *in_tensor.shape[1:],
            dtype=in_tensor.dtype, device=in_tensor.device)
        tensor = torch.cat([in_tensor, zeros])
    else:
        tensor = in_tensor
    # transform the range from [0, 1] to [0, 255]
    tensor = tensor.mul_(255).add_(0.5).clamp_(0, 255)
    # tile the images into a square grid (B1 * H, B2 * W),
    # and rearrange the channels to the last dimension
    tensor = einops.rearrange(
        tensor, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=bsqr, b2=bsqr)
    # compress the images using PIL
    image = Image.fromarray(tensor.to('cpu', torch.uint8).numpy())
    stream = io.BytesIO()
    image.save(stream, 'JPEG', quality=quality, optimice=True)
    stream.seek(0)
    array = np.array(Image.open(stream), copy=True)
    tensor = torch.from_numpy(array).float()
    tensor = tensor.to(in_tensor.device)
    # transform the range from [0, 255] to [0, 1]
    tensor = tensor.sub_(0.5).div_(255).clamp_(0, 1)
    # untile the images to (B, C, H, W)
    tensor = einops.rearrange(
        tensor, '(b1 h) (b2 w) c -> (b1 b2) c h w', b1=bsqr, b2=bsqr)
    # remove the padding
    if bpad:
        tensor = tensor[:-bpad]
    # a surrogate for the gradient of the JPEG compression
    return (in_tensor - in_tensor.detach()) + tensor


class JPEGLayer(torch.nn.Module):
    def __init__(self, quality=None):
        super().__init__()
        self.quality = quality

    def forward(self, tensor):
        return jpeg_compression(tensor, self.quality)


UEraser = K.AugmentationSequential(
    K.RandomPlasmaBrightness(
        roughness=(0.1, 0.7), intensity=(0.0, 1.0),
        same_on_batch=False, p=0.5, keepdim=True),
    # K.RandomPlasmaContrast(roughness=(0.1, 0.7), p=0.5),
    K.RandomChannelShuffle(same_on_batch=False, p=0.5, keepdim=True),
    K.auto.TrivialAugment(),
)
UEraserJPEG =K.AugmentationSequential(
    K.RandomPlasmaBrightness(
        roughness=(0.3, 0.7), intensity=(0.5, 1.0),
        same_on_batch=False, p=0.5, keepdim=True,
    ),
    # K.RandomPlasmaContrast(roughness=(0.3, 0.7), p=0.5),
    K.RandomChannelShuffle(same_on_batch=False, p=0.5, keepdim=True),
    K.auto.TrivialAugment(),
    JPEGLayer(10),
)


class Criterion(Protocol):
    def __call__(
        self, input: Tensor, target: Tensor, reduction: str='mean', **kwargs
    ) -> Tensor: ...


def adversarial_augmentation_loss(
    model: nn.Module, images: Tensor, labels: Tensor,
    repeat: int, criterion: Criterion = torch.nn.functional.cross_entropy,
    augs: torch.nn.Module = UEraser,
) -> Tensor:
    """
    This function computes the adversarial augmentation loss.

    model:
        The model being trained.
    images:
        A batch of images in the range [0, 1] and shape (B, C, H, W),
        where B is the batch size, C is the number of channels,
        H is the height, and W is the width.
    labels:
        A batch of true labels in the range [0, K) and shape (B,),
        where B is the batch size and K is the number of classes.
    repeat:
        The number of times to repeat the augmentation policy for each image.
    criterion:
        The loss function. The default is cross-entropy loss.
    augs:
        The augmentation policy. The default is UEraser.

    Returns:
        The loss value.
    """
    if repeat < 0:
        raise ValueError('The number of repeats must be non-negative.')
    if repeat == 0:
        # disable augmentation
        logits = model(images)
        return criterion(logits, labels)
    if repeat == 1:
        # UEraser-Lite
        images = augs(images)
        logits = model(images)
        return criterion(logits, labels)
    # adversarial augmentation
    # repeat images and labels
    images = images.tile(repeat, *((1, ) * (images.ndim - 1)))
    labels = labels.tile(repeat)
    # pass through different augmentation policies
    aug_images = augs(images)
    # forward pass
    logits = model(aug_images)
    # compute loss
    losses = criterion(logits, labels, reduction='none')
    # find the max loss for all repeated augmentations of each image
    max_losses = torch.max(losses.view(repeat, -1), dim=0).values
    # return the mean of max losses
    return torch.mean(max_losses)
