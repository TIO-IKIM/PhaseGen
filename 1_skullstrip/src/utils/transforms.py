# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import torch


def augmentation(image, mask):
    """
    Augment image and mask data.
    Vertical flip, random rotation, random brightness & contrast.

    Args:
        image: image input data (x).
        mask: Label input data (y).

    Returns:
        Augmented image and label data.
    """

    rotation_param = transforms.RandomRotation.get_params([-5, 5])
    image = TF.rotate(image, rotation_param)
    mask = TF.rotate(mask, rotation_param)

    # brightness_param = random.uniform(0.7, 1)
    # image = TF.adjust_brightness(image, brightness_param)
    # mask = TF.adjust_brightness(mask, brightness_param)

    # image, mask = transforms.ToTensor(image), transforms.ToTensor(mask)

    return image, mask


def periphery_augmentation(
    input: torch.Tensor, label: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complex augmentation by adjusting the periphery of the k-space.
    A mask is created in the range given by the integer `n` around the center.
    The mask is then scaled with the float `s` and multiplied with the input tensor.

    Args:
        input (torch.Tensor): Complex valued input k-space tensor.
        label (torch.Tensor): Corresponding complex valued label k-space tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The complex valued periphery augmented input and label k-space tensor.
    """

    n = random.randint(5, 40)
    s = random.uniform(0.7, 1.3)
    begin = int(input.size(1) / 2 - n)
    end = int(input.size(1) / 2 + n)

    mask = torch.ones(input.size()[1:])
    mask *= s
    mask[begin:end, begin:end] = 1

    input[0, ...] *= mask
    if label is not None:
        label[0, ...] *= mask

        # n = 25
        # s = 1
        # begin = int(input.size(1) / 2 - n)
        # end = int(input.size(1) / 2 + n)

        # mask = torch.ones(input.size()[1:])
        # mask *= s
        # mask[begin:end, begin:end] = 0.5

        # input[0, ...] *= mask
        # label[0, ...] *= mask

        return input, label
    else:
        return input


def positive(x: torch.cfloat):
    x_pos = x.real.abs() + 1j * x.imag.abs()

    return x_pos
