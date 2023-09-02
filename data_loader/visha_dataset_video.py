import os
import torch
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from torch.utils.data import Dataset
from .augmentations_video import get_train_joint_transform, get_val_joint_transform


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
            self.time_clips = config['time_clips']
        else:
            self.joint_transform = get_val_joint_transform(scale=config["scale"])
            self.time_clips = config['time_clips']

        # get all frames from video datasets
        self.frame_list, self.path_image, self.path_mask = self.generate_images_from_video(is_training=self.is_training)
        print('Total video clips are {}.'.format(len(self.frame_list)))

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, index):
        image_label_path_list = self.frame_list[index]

        clip_list = []
        label_list = []
        w_list = []
        h_list = []
        image_path_list = []
        label_path_list = []
        for image_path, label_path in image_label_path_list:
            if not self.is_training:
                image_path = self.path_image[image_path]
                label_path = self.path_mask[label_path]
                image = Image.open(image_path).convert('RGB')
                label = Image.open(label_path).convert('L')
            else:
                image = self.path_image[image_path]
                label = self.path_mask[label_path]
            clip_list.append(image)
            label_list.append(label)
            w, h = image.size
            w_list.append(w)
            h_list.append(h)
            image_path_list.append(image_path)
            label_path_list.append(label_path)

        clip_list, label_list = self.joint_transform(clip_list, label_list)
        image_torch = torch.stack(clip_list)
        label_torch = torch.stack(label_list)

        return {"image": image_torch, "label": label_torch, "image_path": image_path_list, "label_path": label_path_list, "w": w_list, "h": h_list}

    def generate_images_from_video(self, is_training=True):
        video_list = os.listdir(os.path.join(self.data_root, self.mode, self.image_folder))
        video_frame_dict = {}
        path_frame_dict = {}
        path_mask_dict = {}

        for video in video_list:
            video_path = os.path.join(self.data_root, self.mode, self.image_folder, video)
            frame_list = [os.path.splitext(frame)[0] for frame in os.listdir(video_path) if frame.endswith(self.image_ext)]
            frame_list = self.sort_images(frame_list)

            if self.is_training:
                # add more frames in revised order if less than 100 frames
                len_frame_list = len(frame_list)
                if len_frame_list < 100:
                    for _ in range(int(100/len_frame_list)+1):
                        for reversed_frame in frame_list[-1:-(min(100-len_frame_list, len_frame_list)):-1]:
                            frame_list.append(reversed_frame)
                    if len(frame_list) >= 100:
                        frame_list = frame_list[:100]

            video_frame_dict[video] = []
            for frame in frame_list:
                # frame_gt: (frame, gt)
                frame_path = os.path.join(self.data_root, self.mode, self.image_folder, video, frame + self.image_ext)
                gt_path = os.path.join(self.data_root, self.mode, self.label_folder, video, frame + self.label_ext)
                
                frame_gt = (frame_path, gt_path)
                video_frame_dict[video].append(frame_gt)

                # if training, load data in init function in adcance. if testing, load data path for fast preprocessing.
                if is_training:
                    path_frame_dict[frame_path] = Image.open(frame_path).convert('RGB')
                    path_mask_dict[gt_path] = Image.open(gt_path).convert('L')
                else:
                    path_frame_dict[frame_path] = frame_path
                    path_mask_dict[gt_path] = gt_path

        # ensemble clips
        clip_list = []
        for video in video_list:
            frames_from_one_video = video_frame_dict[video]
            stride = 1 if self.is_training else self.time_clips
            for begin in range(0, len(frames_from_one_video) - self.time_clips + 1, stride):
                frame_clips = frames_from_one_video[begin: begin + self.time_clips]
                clip_list.append(frame_clips)

            # last n image go backward for training, and last clip for test
            if self.is_training:
                for begin in range(len(frames_from_one_video) - self.time_clips + 1, len(frames_from_one_video)):
                    frame_clips = frames_from_one_video[begin: begin-self.time_clips: -1]
                    clip_list.append(frame_clips)
            else:
                last_frame_clips = frames_from_one_video[len(frames_from_one_video) - self.time_clips:]
                clip_list.append(last_frame_clips)

        return clip_list, path_frame_dict, path_mask_dict

    def sort_images(self, frame_list):
        frame_int_list = [int(frame) for frame in frame_list]
        # sort images to 001, 002, 003...
        sort_index = [i for i, v in sorted(enumerate(frame_int_list), key=lambda x: x[1])]
        return [frame_list[i] for i in sort_index]

    def read_segmentation_mask(self, gt_path):
        gt_pil = Image.open(gt_path).convert('L')
        gt_np = np.array(gt_pil)

        # some gt are store in RGB, whose values are not [0, 255]
        if len(np.unique(gt_np)) != 2:
            gt_np[gt_np != 0] = 255

        return Image.fromarray(gt_np)
