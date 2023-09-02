import random
import numpy as np
from PIL import Image
from torchvision import transforms


def get_train_joint_transform(scale=(512, 512)):
    joint_transform = Compose([
        Resize(scale),
        RandomHorizontallyFlip(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return joint_transform


def get_val_joint_transform(scale=(512, 512)):
    joint_transform = Compose([
        Resize(scale),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return joint_transform
    

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs, masks):
        assert len(imgs) == len(masks)
        for t in self.transforms:
            imgs, masks = t(imgs, masks)
        return imgs, masks


class RandomCropAndResize(object):
    def __init__(self, scale=(256, 488)):
        assert scale[0] <= scale[1]
        self.scale = scale
        self.crop = RandomCrop(ratio=(0.0546875, 0.03125))

    def __call__(self, imgs, masks):
        imgs, masks = self.crop(imgs, masks)
        w, h = imgs[0].size

        for idx in range(len(imgs)):
            if w > h:
                img, mask = imgs[idx].resize((self.scale[1], self.scale[0]), Image.BILINEAR), masks[idx].resize((self.scale[1], self.scale[0]), Image.NEAREST)
            else:
                img, mask = imgs[idx].resize((self.scale[0], self.scale[1]), Image.BILINEAR), masks[idx].resize((self.scale[0], self.scale[1]), Image.NEAREST)
            imgs[idx] = img
            masks[idx] = mask

        return imgs, masks


class RandomCrop(object):
    def __init__(self, ratio=(0.0546875, 0.03125)):
        # ratio[0] is for short edge, ratio[1] is for long edge
        self.ratio=ratio

    def __call__(self, imgs, masks):
        w, h = imgs[0].size
        if w > h:
            self.ratio = (self.ratio[1], self.ratio[0])

        tw, th = int(w * (1-self.ratio[0])), int(h * (1-self.ratio[1]))

        x, y = random.randint(0, w - tw), random.randint(0, h - th)

        for idx in range(len(imgs)):
            imgs[idx], masks[idx] = imgs[idx].crop((x, y, x + tw, y + th)), masks[idx].crop((x, y, x + tw, y + th))

        return imgs, masks


class RandomHorizontallyFlip(object):
    def __call__(self, imgs, masks):
        if random.random() < 0.5:
            for idx in range(len(imgs)):
                imgs[idx] = imgs[idx].transpose(Image.FLIP_LEFT_RIGHT)
                masks[idx] = masks[idx].transpose(Image.FLIP_LEFT_RIGHT)

        return imgs, masks
        

class Resize(object):
    def __init__(self, scale):
        assert scale[0] <= scale[1]
        self.scale = scale

    def __call__(self, imgs, masks):
        w, h = imgs[0].size
        for idx in range(len(imgs)):
            if w > h:
                imgs[idx] = imgs[idx].resize((self.scale[1], self.scale[0]), Image.BILINEAR)
                masks[idx] = masks[idx].resize((self.scale[1], self.scale[0]), Image.NEAREST)
            else:
                imgs[idx] = imgs[idx].resize((self.scale[0], self.scale[1]), Image.BILINEAR)
                masks[idx] = masks[idx].resize((self.scale[0], self.scale[1]), Image.NEAREST)

        return imgs, masks


class ToTensor(object):
    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, imgs, masks):
        for idx in range(len(imgs)):
            img_np = np.array(imgs[idx])
            mask_np = np.array(masks[idx])
            
            x, y, _ = img_np.shape
            # make sure x is short than y
            if x > y:
                img_np = np.swapaxes(img_np, 0, 1)
                mask_np = np.swapaxes(mask_np, 0, 1)

            imgs[idx] = self.totensor(img_np)
            masks[idx] = self.totensor(mask_np).long()

        return imgs, masks


class Normalize(object):
    def __init__(self, mean, std):
        self.normlize = transforms.Normalize(mean, std)

    def __call__(self, imgs, masks):
        for idx in range(len(imgs)):
            imgs[idx] = self.normlize(imgs[idx])
        return imgs, masks
