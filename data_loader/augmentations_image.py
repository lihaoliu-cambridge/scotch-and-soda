import random
import numpy as np
from PIL import Image


def get_train_joint_transform(scale=(512, 512)):
    joint_transform = Compose([
        Resize(scale),
        RandomHorizontallyFlip()
    ])
    return joint_transform


def get_val_joint_transform(scale=(512, 512)):
    joint_transform = Compose([
        Resize((scale)),
    ])
    return joint_transform
    

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs, masks):
        assert imgs.size == masks.size
        for t in self.transforms:
            imgs, masks = t(imgs, masks)
        return imgs, masks


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, imgs, masks):
        assert imgs.size == masks.size
        return imgs.resize(self.size, Image.BILINEAR), masks.resize(self.size, Image.BILINEAR)


class RandomHorizontallyFlip(object):
    def __call__(self, imgs, masks):
        if random.random() < 0.5:
            return imgs.transpose(Image.FLIP_LEFT_RIGHT), masks.transpose(Image.FLIP_LEFT_RIGHT)
        return imgs, masks
