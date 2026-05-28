import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.setup_utils import get_subject_tag


class SmoothedNLLLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 1:
            target = F.one_hot(target.long(), num_classes=self.num_classes).float()
        else:
            target = target.float()

        if self.smoothing > 0:
            target = target * (1.0 - self.smoothing) + self.smoothing / self.num_classes

        return -(target * log_probs).sum(dim=1).mean()


def get_criterion(args):
    return SmoothedNLLLoss(
        num_classes=args.num_classes,
        smoothing=float(getattr(args, "smoothing", 0.0)),
    )


def get_optimizer(model, args):
    return optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.EPOCHS,
        eta_min=0,
    )


def get_checkpoint_callback(fold: int, monitor: str, args):
    dirpath = f"{args.CKPT_PATH}/{args.LOG_NAME}/fold_{fold + 1}"
    subject_tag = get_subject_tag(args.target_subject)

    if monitor == "val_acc":
        return ModelCheckpoint(
            monitor=monitor,
            dirpath=dirpath,
            filename=f"{args.task}_{subject_tag}" + "_{epoch:02d}-{val_acc:.3f}",
            save_top_k=1,
            mode="max",
        )

    if monitor == "val_loss":
        return ModelCheckpoint(
            monitor=monitor,
            dirpath=dirpath,
            filename=f"{args.task}_{subject_tag}" + "_{epoch:02d}-{val_loss:.3f}",
            save_top_k=1,
            mode="min",
        )

    return None


def get_callbacks(fold: int, monitor: str, args):
    callbacks = []
    checkpoint_callback = get_checkpoint_callback(fold=fold, monitor=monitor, args=args)
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    return callbacks
