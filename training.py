import argparse
import os
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler

from model.litmodel import get_litmodel
from preprocessing.bci_compet import get_dataset
from utils.setup_utils import clone_args, get_device, get_log_name, get_subject_tag, load_config
from utils.training_utils import get_callbacks


parser = argparse.ArgumentParser()
parser.add_argument("--subject_num", type=int, default=-1)
parser.add_argument("--fold_num", type=int, default=-1)
parser.add_argument("--gpu_num", type=str, default="0")
parser.add_argument("--config_name", type=str, default="bcicompet2b_config")
cli_args = parser.parse_args()

args = load_config(cli_args.config_name)
seed_everything(args.SEED)

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = cli_args.gpu_num
args["device"] = get_device(cli_args.gpu_num)

cudnn.benchmark = torch.cuda.is_available()
cudnn.deterministic = True

args["current_time"] = datetime.now().strftime("%Y%m%d")
args["LOG_NAME"] = get_log_name(args)
args.lr = float(args.lr)
if args.downsampling:
    args["sampling_rate"] = args.downsampling

trainer_kwargs = {
    "enable_progress_bar": True,
    "max_epochs": args.EPOCHS,
    "default_root_dir": args.CKPT_PATH,
}
if torch.cuda.is_available():
    trainer_kwargs.update({"accelerator": "gpu", "devices": 1})
else:
    trainer_kwargs.update({"accelerator": "cpu", "devices": 1})

base_dataset_args = clone_args(args, is_test=False)

for num_subject in range(args.num_subjects):
    if cli_args.subject_num != -1 and num_subject != cli_args.subject_num:
        continue

    subject_tag = get_subject_tag(num_subject)
    train_args = clone_args(base_dataset_args, target_subject=num_subject, use_augmentation=True)
    val_args = clone_args(base_dataset_args, target_subject=num_subject, use_augmentation=False)

    train_dataset = get_dataset(args.config_name, train_args)
    val_dataset = get_dataset(args.config_name, val_args)

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.SEED)

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(range(len(train_dataset))), start=1):
        if cli_args.fold_num != -1 and fold_idx != cli_args.fold_num:
            continue

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            pin_memory=False,
            num_workers=args.num_workers,
            sampler=SubsetRandomSampler(train_idx),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            pin_memory=False,
            num_workers=args.num_workers,
            sampler=SubsetRandomSampler(val_idx),
        )

        model_args = clone_args(args, target_subject=num_subject)
        model = get_litmodel(model_args)
        logger = TensorBoardLogger(
            save_dir=args.LOG_PATH,
            name=args.LOG_NAME,
            version=f"{subject_tag}_fold{fold_idx}",
        )
        callbacks = get_callbacks(fold=fold_idx - 1, monitor="val_acc", args=model_args)
        trainer = Trainer(logger=logger, callbacks=callbacks, **trainer_kwargs)

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
