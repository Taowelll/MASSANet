import torch
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy

from model.MASSANet import get_model
from utils.training_utils import get_criterion, get_optimizer, get_scheduler


class LitModel(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = get_model(args)
        self.criterion = get_criterion(args)

    def forward(self, x):
        return self.model(x.float())

    @staticmethod
    def _to_hard_labels(labels):
        if labels.dim() == 1:
            return labels.long()
        return torch.argmax(labels, dim=1).long()

    def _shared_step(self, batch, stage: str):
        inputs = batch["data"].float()
        labels = batch["label"]
        hard_labels = self._to_hard_labels(labels)

        log_probs = self(inputs)
        loss = self.criterion(log_probs, labels)
        preds = torch.argmax(log_probs, dim=1)
        acc = accuracy(
            preds,
            hard_labels,
            task="multiclass",
            num_classes=self.args.num_classes,
        )

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=(stage != "train"))
        self.log(f"{stage}_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch["data"].float())

    def configure_optimizers(self):
        optimizer = get_optimizer(self, self.args)
        scheduler = get_scheduler(optimizer, self.args)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def get_litmodel(args):
    return LitModel(args)
