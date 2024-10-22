import wandb
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
from pyro.distributions.transforms.planar import Planar
from pyro.distributions.transforms.radial import Radial
from pyro.distributions.transforms.affine_autoregressive import AffineAutoregressive, affine_autoregressive
from torch import nn
import torch.distributions as tdist
from torch.nn.utils import spectral_norm, remove_spectral_norm
import pytorch_lightning as pl


class PrintValAccuracyCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # Access the current validation accuracy from the logged metrics
        val_acc = trainer.callback_metrics.get('val_accuracy')
        if val_acc is not None:
            # Print the current validation accuracy
            print(f"Current Validation Accuracy: {val_acc:.4f}")

        # If you're using ModelCheckpoint and want to print info related to the best val_acc
        # Find the ModelCheckpoint callback from the trainer's callbacks list
        checkpoint_callback = next((cb for cb in trainer.callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint)), None)
        if checkpoint_callback:
            # Access the best validation accuracy recorded by the ModelCheckpoint callback
            best_val_acc = checkpoint_callback.best_model_score
            if best_val_acc is not None:
                print(f"Best Validation Accuracy so far: {best_val_acc:.4f}")

class DefineMetricCallback_cls(Callback):
    def setup(self, trainer, pl_module, stage):
        # Check if this is the main process
        if trainer.is_global_zero:
            wandb.define_metric("val_accuracy", summary="max")
            wandb.define_metric("val_loss", summary="min")

class DefineMetricCallback_reg(Callback):
    def setup(self, trainer, pl_module, stage):
        # Check if this is the main process
        if trainer.is_global_zero:
            wandb.define_metric("val_L1loss", summary="min")
            wandb.define_metric("val_loss", summary="min")
