import argparse
import os
import re
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer, seed_everything
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader

from model.litmodel import get_litmodel
from preprocessing.bci_compet import get_dataset
from utils.setup_utils import (
    clone_args,
    get_device,
    get_subject_label,
    get_subject_tag,
    load_config,
)


def select_checkpoint(ckpt_list):
    if len(ckpt_list) == 1:
        return ckpt_list[0]

    def score(path):
        match = re.search(r"-(\d+\.\d+)\.ckpt$", path)
        return float(match.group(1)) if match else float("-inf")

    return max(ckpt_list, key=score)


parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str, default="bcicompet2b_config")
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--subject_num", type=int, default=-1)
cli_args = parser.parse_args()

args = load_config(cli_args.config_name)
args["current_time"] = datetime.now().strftime("%Y%m%d")
args["LOG_NAME"] = cli_args.ckpt_path

if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_NUM
args["device"] = get_device(args.GPU_NUM)

cudnn.benchmark = torch.cuda.is_available()
cudnn.deterministic = True
seed_everything(args.SEED)

if args.downsampling:
    args["sampling_rate"] = args.downsampling

trainer_kwargs = {}
if torch.cuda.is_available():
    trainer_kwargs.update({"accelerator": "gpu", "devices": 1})
else:
    trainer_kwargs.update({"accelerator": "cpu", "devices": 1})

base_eval_args = clone_args(args, is_test=True, use_augmentation=False)

total_results = []
total_kappas = []
total_f1_macro = []
total_f1_weighted = []
evaluated_subjects = []

for num_subject in range(args.num_subjects):
    if cli_args.subject_num != -1 and num_subject != cli_args.subject_num:
        continue

    subject_tag = get_subject_tag(num_subject)
    subject_label = get_subject_label(num_subject)
    evaluated_subjects.append(subject_label)
    eval_args = clone_args(base_eval_args, target_subject=num_subject)

    print(f"\n{'=' * 50}")
    print(f"Evaluating Subject {subject_label}")
    print(f"{'=' * 50}")

    dataset = get_dataset(args.config_name, eval_args)
    logits_sum = np.zeros((len(dataset), args.num_classes), dtype=np.float32)
    valid_fold_count = 0

    test_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        num_workers=args.num_workers,
        shuffle=False,
    )

    for fold_idx in range(1, args.k_folds + 1):
        ckpt_list = sorted(glob(f"{args.CKPT_PATH}/{args.LOG_NAME}/fold_{fold_idx}/*{subject_tag}*.ckpt"))
        if len(ckpt_list) == 0:
            print(f"[WARN] Missing checkpoint: {subject_label}, Fold {fold_idx}")
            continue

        ckpt_path = select_checkpoint(ckpt_list)
        print(f"[Eval] Using checkpoint: {ckpt_path}")

        model = get_litmodel(eval_args)
        state_dict = torch.load(ckpt_path, map_location=args.device)["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        trainer = Trainer(**trainer_kwargs)
        logits = trainer.predict(model, dataloaders=test_dataloader)
        logits_sum += torch.cat(logits, dim=0).detach().cpu().numpy()
        valid_fold_count += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if valid_fold_count == 0:
        print(f"[WARN] No valid checkpoints found for {subject_label}. Skipped.")
        evaluated_subjects.pop()
        continue

    logits_sum /= valid_fold_count
    y_pred = logits_sum.argmax(axis=1)
    y_true = np.asarray(dataset.label)

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    total_results.append(acc)
    total_kappas.append(kappa)
    total_f1_macro.append(f1_macro)
    total_f1_weighted.append(f1_weighted)

    print(f"\n{subject_label} Results:")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Kappa       : {kappa:.4f}")
    print(f"F1-macro    : {f1_macro:.4f}")
    print(f"F1-weighted : {f1_weighted:.4f}")

result_df = pd.DataFrame(
    {
        "Acc.": total_results,
        "Kappa": total_kappas,
        "F1-macro": total_f1_macro,
        "F1-weighted": total_f1_weighted,
    },
    index=evaluated_subjects,
)

result_df.loc["Avg."] = result_df.mean()
result_df.loc["Std."] = result_df.std(ddof=0)

print("\n\n")
print("=" * 70)
print("=" * 27, " Result ", "=" * 27)
print(result_df)
print("=" * 70)
print(f"Mean Accuracy    : {np.mean(total_results) * 100:.2f}%")
print(f"Std Accuracy     : {np.std(total_results, ddof=0) * 100:.2f}%")
print(f"Mean Kappa       : {np.mean(total_kappas):.4f}")
print(f"Std Kappa        : {np.std(total_kappas, ddof=0):.4f}")
print(f"Mean F1-macro    : {np.mean(total_f1_macro):.4f}")
print(f"Std F1-macro     : {np.std(total_f1_macro, ddof=0):.4f}")
print(f"Mean F1-weighted : {np.mean(total_f1_weighted):.4f}")
print(f"Std F1-weighted  : {np.std(total_f1_weighted, ddof=0):.4f}")
print("-" * 70)
print(f"Reported Accuracy    : {np.mean(total_results) * 100:.2f} +- {np.std(total_results, ddof=0) * 100:.2f}%")
print(f"Reported Kappa       : {np.mean(total_kappas):.4f} +- {np.std(total_kappas, ddof=0):.4f}")
print(f"Reported F1-macro    : {np.mean(total_f1_macro):.4f} +- {np.std(total_f1_macro, ddof=0):.4f}")
print(f"Reported F1-weighted : {np.mean(total_f1_weighted):.4f} +- {np.std(total_f1_weighted, ddof=0):.4f}")
print("=" * 70)
print("\n\n")
