import os
import torch
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from .augmentations_image import get_train_joint_transform, get_val_joint_transform


class ViSha_Dataset(Dataset):
    def __init__(self, mode: str, config: dict) -> None:
        self.config = config
        self.mode = mode
        self.is_training = (mode == "train")
        print("Dataloader Mode:", "training" if self.is_training else "testing")

        # configs
        self.data_root = config["data_root"]
        self.image_folder = config['image_folder']
        self.label_folder = config['label_folder']
        self.image_ext = config['image_ext']
        self.label_ext = config['label_ext']

        # transform
        if self.is_training:
            self.joint_transform = get_train_joint_transform(scale=config["scale"]) 
        else:
            self.joint_transform = get_val_joint_transform(scale=config["scale"])

        # # original code with mobilenet:
        # self.to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        # self.to_tensor_transform_target = transforms.Compose([transforms.ToTensor()])
        # modified code with imagenet (different mean and std):
        self.to_tensor_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.to_tensor_transform_target = transforms.Compose([transforms.ToTensor()])

        # get all frames from video datasets
        self.image_gt_list = self.make_visha(is_train=self.is_training)
        print('Total video clips are {}.'.format(len(self.image_gt_list)))

    def __len__(self):
        return len(self.image_gt_list) + (len(self.image_gt_list) % int(self.config["time_clips"]))

    def __getitem__(self, index):
        img_path, gt_path = self.image_gt_list[index % len(self.image_gt_list)]

        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        target = Image.open(gt_path).convert('L')

        img, target = self.joint_transform(img, target)
        img_torch = self.to_tensor_transform(img)
        target_torch = self.to_tensor_transform_target(target)

        return {"image": img_torch, "label": target_torch, "image_path": img_path, "label_path": gt_path, "w": w, "h": h}

    def make_visha(self, is_train=True):
        video_root = os.path.join(self.data_root, 'train' if is_train else 'test')

        video_list = [video for video in os.listdir(os.path.join(video_root, self.image_folder)) if os.path.isdir(os.path.join(video_root, self.image_folder, video))]
        img_name = []
        for v in video_list:
            for img in os.listdir(os.path.join(video_root, self.image_folder, v)):
                if img.endswith('.jpg'):
                    img_name.append([os.path.join(video_root, self.image_folder, v, img), os.path.join(video_root, self.label_folder, v, img[:-4]+'.png')])
        return img_name   
