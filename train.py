import os
import yaml
import pytorch_lightning as pl
from typing import List
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

pl.seed_everything(1234)


def load_config(config_path: str, config_name: str) -> List[dict]:
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    dataset_config = config["dataset"]
    model_config = config["model"]
    trainer_config = config["train"]

    return dataset_config, model_config, trainer_config


def prepare_dataloader(dataset_config: dict, batch_size: int = 1, mode: str = "train"):
    assert mode in ["train", "test"]

    if dataset_config["dataset_name"] == "ViSha_dataset_image":
        from data_loader.visha_dataset_image import ViSha_Dataset
    elif dataset_config["dataset_name"] == "ViSha_dataset_video":
        from data_loader.visha_dataset_video import ViSha_Dataset
    else:
        raise NotImplementedError("No dataset {}".format(dataset_config["dataset_name"]))
    
    dataset = ViSha_Dataset

    visha_dataset = dataset(mode, dataset_config)  # type: ignore

    if mode == "train":
        dataloader = DataLoader(visha_dataset, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(visha_dataset, batch_size=batch_size)

    return dataloader


def prepare_model(model_config: dict, is_train=True) -> pl.LightningModule:
    if model_config["model_name"] == "scotch_and_soda":
        from model.scotch_and_soda import LightningNetwork
    else:
        raise NotImplementedError("No model {}".format(model_config["model_name"]))

    lightning_model = LightningNetwork(model_config)

    if (model_config["pretrained_weight"] != None) and is_train:
        model = lightning_model.load_from_checkpoint(
            checkpoint_path=model_config["pretrained_weight"],
            configs=model_config,
            strict=False
        )
    else:
        model = lightning_model
    
    return model


def prepare_pl_trainer(trainer_config: dict) -> pl.Trainer:
    # tensorboard logger and learning rate monitor
    tb_dir = os.path.join(trainer_config["output_dir"], trainer_config["tb_dirname"])
    tb_logger = pl.loggers.TensorBoardLogger(tb_dir, name=trainer_config["experiment_name"], default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(trainer_config["output_dir"], trainer_config["checkpoint"]["ckpt_dirname"], trainer_config["experiment_name"]),
        save_top_k=-1, # -1 means save all models
        save_weights_only=True
    )

    pl_trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        # basic config: gpus, epochs, output_dir
        gpus=trainer_config["gpus"],
        max_epochs=trainer_config["max_epochs"],
        default_root_dir=trainer_config["output_dir"],
        log_every_n_steps=trainer_config["log_every_n_steps"],
        # gpu accelerate
        accelerator=trainer_config["accelerator"],
        strategy=trainer_config["strategy"],
    )

    return pl_trainer


def main():
    # args and config
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./config/")
    parser.add_argument('--config_name', type=str, default="scotch_and_soda_visha_image_config.yaml")
    args = parser.parse_args()

    dataset_config, model_config, trainer_config = load_config(args.config_path, args.config_name)

    # prepare dataloader
    training_dataloader = prepare_dataloader(dataset_config, batch_size=trainer_config["batch_size"])

    # prepare model
    model = prepare_model(model_config)
    
    # prepare training pipeline
    pl_trainer = prepare_pl_trainer(trainer_config)

    # training
    pl_trainer.fit(model, training_dataloader)


if __name__ == '__main__':
    main()
