import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  
import pytorch_lightning as pl
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation
from transformers.modeling_outputs import SemanticSegmenterOutput
from einops import rearrange
import matplotlib.pyplot as plt
from medpy import metric

from model.layer.deformation_trajectory_attention import DeformationTrajectoryAttentionBlock
                    

class Projection(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model(x)
        return F.normalize(x, dim=1)


class ScotchAndSoda(nn.Module):
    def __init__(self, frame_size):
        super().__init__()
        self.pretrained_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512", num_labels=1, ignore_mismatched_sizes=True)
        self.config = self.pretrained_model.config
        self.segformer = self.pretrained_model.segformer
        self.decode_head = self.pretrained_model.decode_head

        self.dims = [64, 128, 320, 512]
        self.nums_heads_list=[8, 8, 8, 8]
        self.x_sizes = [128, 64, 32, 16]
        self.y_sizes = [128, 64, 32, 16]
        self.frame_sizes=[frame_size] * 4

        self.deformation_trajectory_attention_1 = DeformationTrajectoryAttentionBlock(dim=self.dims[1], num_heads=self.nums_heads_list[1], x_size=self.x_sizes[1], y_size=self.y_sizes[1], frame_size=self.frame_sizes[1])
        self.deformation_trajectory_attention_2 = DeformationTrajectoryAttentionBlock(dim=self.dims[2], num_heads=self.nums_heads_list[2], x_size=self.x_sizes[2], y_size=self.y_sizes[2], frame_size=self.frame_sizes[2])
        self.deformation_trajectory_attention_3 = DeformationTrajectoryAttentionBlock(dim=self.dims[3], num_heads=self.nums_heads_list[3], x_size=self.x_sizes[3], y_size=self.y_sizes[3], frame_size=self.frame_sizes[3])
        self.attns_list = [None, self.deformation_trajectory_attention_1, self.deformation_trajectory_attention_2, self.deformation_trajectory_attention_3]

    def forward(self, pixel_values, labels = None, output_attentions = None, output_hidden_states = True, return_dict = None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.segformer(
            pixel_values,
            output_attentions=True,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        
        attended_hidden_states = [encoder_hidden_states[0]]

        for i in range(1, len(encoder_hidden_states)):
            hidden_states = encoder_hidden_states[i]
            attn_outputs = self.attns_list[i](hidden_states)
            attn_hidden_states = attn_outputs[0]

            attended_hidden_states.append(attn_hidden_states)

        logits = self.decode_head(attended_hidden_states)

        loss = None

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


class LightningNetwork(pl.LightningModule):
    def __init__(self, configs: dict):
        super().__init__()
        self.backbone = ScotchAndSoda(configs['time_clips'])
        self.projection = Projection()

        self.experiment_name = configs["experiment_name"]
        self.output_dir = configs["output_dir"]

        self.loss_type = configs["loss_type"]
        self.finetune_learning_rate = configs["finetune_learning_rate"]
        self.scratch_learning_rate = configs["scratch_learning_rate"]
        self.optimizer = configs["optimizer"]
        self.lr_scheduler = configs["lr_scheduler"]

        self.max_epochs = configs["max_epochs"]
        self.time_clips = configs["time_clips"]

        if self.loss_type == "l1":
            self.segmentation_loss = nn.L1Loss()
        elif self.loss_type == "bce":            
            self.segmentation_loss = nn.BCELoss()
        elif self.loss_type == "mixed":
            from util.loss.losses import lovasz_hinge, binary_xloss, scotch_loss
            self.segmentation_loss = binary_xloss
            self.lovasz_hinge = lovasz_hinge
            self.contrastive_loss = scotch_loss
        else:
            raise Exception

        self.save_hyperparameters()

        self.to_pil = transforms.ToPILImage()

    def forward(self, x: torch.Tensor):  # type: ignore
        print("Didn't implement this forward function.")

    def training_step(self, batch: dict, batch_idx: int):  # type: ignore
        exemplar, labels = batch['image'], batch['label']
        if len(exemplar.size()) == 5 and len(labels.size()) == 5:
            exemplar = exemplar.flatten(start_dim=0, end_dim=1).contiguous() 
            labels = labels.flatten(start_dim=0, end_dim=1).contiguous() 

        outputs = self.backbone(exemplar)
        logits = outputs.logits
        final_outputs = torch.nn.functional.interpolate(
            logits, size=exemplar.size()[2:], mode="bilinear", align_corners=False
        )

        if self.loss_type == "l1":
            final_outputs = F.sigmoid(final_outputs)
            total_loss = self.segmentation_loss(final_outputs, labels)
        elif self.loss_type == "bce":          
            final_outputs = F.sigmoid(final_outputs)  
            total_loss = self.segmentation_loss(final_outputs, labels.float())
        elif self.loss_type == "mixed":          
            segmentation_loss = self.segmentation_loss(final_outputs, labels)
            hinge_loss = self.lovasz_hinge(final_outputs, labels)

            last_hidden_states = outputs.hidden_states[-1]
            mask_shadow = torch.nn.functional.interpolate(labels.float(), size=last_hidden_states.size()[2:], mode="nearest")
            mask_non_shadow = 1 - mask_shadow
            z1 = self.projection(last_hidden_states * mask_shadow)
            z2 = self.projection(last_hidden_states * mask_non_shadow)
            contrastive_loss = self.contrastive_loss(z1, z2)
            total_loss = segmentation_loss + hinge_loss + 0.1 * contrastive_loss
        else:
            raise Exception

        # log loss and images to tensorboard
        tensorboard = self.logger.experiment
        tensorboard.add_scalar("loss", total_loss, self.global_step)
        logged_images = {
            "images/clip_1": (self.reverse_normalize(exemplar[0]).cpu().numpy()*255).astype(np.uint8),
            "images/clip_2": (self.reverse_normalize(exemplar[1]).cpu().numpy()*255).astype(np.uint8),
            "images/clip_3": (self.reverse_normalize(exemplar[2]).cpu().numpy()*255).astype(np.uint8),
            "preds/clip_1":  (final_outputs[0] > 0.5).to(torch.int8),
            "preds/clip_2":  (final_outputs[1] > 0.5).to(torch.int8),
            "preds/clip_3":  (final_outputs[2] > 0.5).to(torch.int8),
            "labels/clip_1": labels[0].unsqueeze(0) if len(labels[0].size()) == 2 else labels[0],
            "labels/clip_2": labels[1].unsqueeze(0) if len(labels[1].size()) == 2 else labels[1],
            "labels/clip_3": labels[2].unsqueeze(0) if len(labels[2].size()) == 2 else labels[2],
            }
        for image_name, image in logged_images.items():
            tensorboard.add_image("{}".format(image_name), image, self.global_step) # self.current_epoch)

        return total_loss

    def test_step(self, batch: dict, batch_idx: int):  # type: ignore
        exemplar, exemplar_gt = batch['image'], batch['label']
        img_path_batch, label_path_batch, w_batch, h_batch = batch["image_path"], batch["label_path"], batch["w"], batch["h"]
        
        is_5d = False
        if len(exemplar.size()) == 5 and len(exemplar_gt.size()) == 5:
            is_5d = True
            exemplar = exemplar.view(-1, *exemplar.size()[2:])
            exemplar_gt = exemplar_gt.view(-1, *exemplar_gt.size()[2:])
            # img_path_batch -> [Time, Batch, 1]

        outputs = self.backbone(exemplar)
        logits = outputs.logits

        final_outputs = torch.nn.functional.interpolate(
            logits, size=exemplar.size()[2:], mode="bilinear", align_corners=False
        )
        exemplar_pre = F.sigmoid(final_outputs)
        for batch_clip_idx in range(0, exemplar_pre.shape[0]):
            prediction = exemplar_pre[batch_clip_idx]
            ground_truth = exemplar_gt[batch_clip_idx]

            if is_5d:
                batch_num = int(batch_clip_idx / self.time_clips)
                clip_num = int(batch_clip_idx % self.time_clips)
                img_path, _, w, h = img_path_batch[clip_num][batch_num], label_path_batch[clip_num][batch_num], w_batch[clip_num][batch_num], h_batch[clip_num][batch_num]
            else:
                batch_idx = batch_clip_idx
                img_path, _, w, h = img_path_batch[batch_idx], label_path_batch[batch_idx], w_batch[batch_idx], h_batch[batch_idx]
            
            self.save_pred_with_exact_value(prediction, ground_truth, img_path, w, h)

    def test_epoch_end(self, outs):
        gt_dir = "./dataset/ViSha/test/labels" # [You might need to change to GT path here if you put your data in a different location]
        pred_dir =  os.path.abspath(os.path.join(self.output_dir, "results", self.experiment_name, 'pred'))

        return computeBER_mth(gt_dir, pred_dir)

    def configure_optimizers(self) -> torch.optim.Adam:
        params = [
            {"params": self.backbone.segformer.parameters(), "lr": self.finetune_learning_rate},
            {"params": self.backbone.decode_head.parameters(), "lr": self.finetune_learning_rate},
            {"params": self.backbone.deformation_trajectory_attention_1.parameters(), "lr": self.scratch_learning_rate},
            {"params": self.backbone.deformation_trajectory_attention_2.parameters(), "lr": self.scratch_learning_rate},
            {"params": self.backbone.deformation_trajectory_attention_3.parameters(), "lr": self.scratch_learning_rate},
            {"params": self.projection.parameters(), "lr": self.scratch_learning_rate},
        ]

        # optimizer
        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(params)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=5e-4, nesterov=False)
        else:
            raise Exception
        
        # # lr_scheduler
        # lambda1 = lambda epoch: (1 - float(epoch) / self.max_epochs) **  0.9
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        # return [optimizer], [scheduler]

        return optimizer

    def log_image(self, image_dict):
        tensorboard = self.logger.experiment
        for image_name, image in image_dict.items():
            tensorboard.add_image("{}".format(image_name), image)

    def reverse_normalize(self, normalized_image):
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        inv_normalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        inv_tensor = inv_normalize(normalized_image)
        return inv_tensor 

    def save_pred_with_exact_value(self, prediction, ground_truth, img_path, w, h, ):
        # # Un-denoted this line and denote the next line, if you wanna binary segmemtation results. 
        # # Empirically, non-binary results has better performance with customed threshold.
        # pred = transforms.Resize((h, w))(self.to_pil((prediction.data.squeeze(0).cpu() > 0.5).float()))
        pred = transforms.Resize((h, w))(self.to_pil(prediction.data.squeeze(0).cpu()))

        sub_name = img_path.split('/')

        check_mkdir(os.path.join(self.output_dir, "results", self.experiment_name, 'pred', sub_name[-2]))
        
        pred.save(os.path.join(self.output_dir, "results", self.experiment_name, 'pred', sub_name[-2], sub_name[-1]))


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_list(dir):
    images = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)

                subname = path.split('/')
                images.append(os.path.join(subname[-2],subname[-1]))
    return images

def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure

def computeBER_mth(gt_path, pred_path): 
    print(gt_path, pred_path)

    gt_list = get_image_list(gt_path)[:100]
    nim = len(gt_list)

    stats = np.zeros((nim, 4), dtype='float')
    stats_jaccard = np.zeros(nim, dtype='float')
    stats_mae = np.zeros(nim, dtype='float')
    stats_fscore = np.zeros((256, nim, 2), dtype='float')

    for i in tqdm(range(0, len(gt_list)), desc="Calculating Metrics:"):
        im = gt_list[i]
        GTim = np.asarray(Image.open(os.path.join(gt_path, im)).convert('L'))
        posPoints = GTim > 0.5
        negPoints = GTim <= 0.5
        countPos = np.sum(posPoints.astype('uint8'))
        countNeg = np.sum(negPoints.astype('uint8'))
        sz = GTim.shape
        GTim = GTim > 0.5

        Predim = np.asarray(Image.open(os.path.join(pred_path, im.replace(".png", ".jpg"))).convert('L').resize((sz[1], sz[0]), Image.NEAREST))
        
        # BER 
        tp = (Predim > 102) & posPoints
        tn = (Predim <= 102) & negPoints
        countTP = np.sum(tp)
        countTN = np.sum(tn)
        stats[i, :] = [countTP, countTN, countPos, countNeg]

        # IoU
        pred_iou = (Predim > 102)
        stats_jaccard[i] = metric.binary.jc(pred_iou, posPoints)

        # MAE
        pred_mae = (Predim > 12)
        mae_value = np.mean(np.abs(pred_mae.astype(float) - posPoints.astype(float)))
        stats_mae[i] = mae_value

        # Precision and Recall for FMeasure
        eps = 1e-4
        for jj in range(0, 256):
            real_tp = np.sum((Predim > jj) & posPoints)
            real_t = countPos
            real_p = np.sum((Predim > jj).astype('uint8'))

            precision_value = (real_tp + eps) / (real_p + eps)
            recall_value = (real_tp + eps) / (real_t + eps)
            stats_fscore[jj, i, :] = [precision_value, recall_value]

    # Print BER 
    posAcc = np.sum(stats[:, 0]) / np.sum(stats[:, 2])
    negAcc = np.sum(stats[:, 1]) / np.sum(stats[:, 3])
    pA = 100 - 100 * posAcc
    nA = 100 - 100 * negAcc
    BER = 0.5 * (2 - posAcc - negAcc) * 100
    print('BER, S-BER, N-BER:')
    print(BER, pA, nA)

    # Print IoU
    jaccard_value = np.mean(stats_jaccard)
    print('IoU:', jaccard_value)

    # Print MAE
    mean_mae_value = np.mean(stats_mae)
    print('MAE:', mean_mae_value)

    # Print Fmeasure
    precision_threshold_list = np.mean(stats_fscore[:, :, 0], axis=1).tolist()
    recall_threshold_list = np.mean(stats_fscore[:, :, 1], axis=1).tolist()
    fmeasure = cal_fmeasure(precision_threshold_list, recall_threshold_list)
    print('Fmeasure:', fmeasure)

    return {"BER": BER, "S-BER": pA, "N-BER": nA, "IoU": jaccard_value, "MAE": mean_mae_value, "Fmeasure": fmeasure}


