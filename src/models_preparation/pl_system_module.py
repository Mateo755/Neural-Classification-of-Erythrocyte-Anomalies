import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix
import wandb
import os

from src.utils.visualization import save_confusion_matrix

class MalariaClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, weight_decay=0.0, num_classes=2):
        super().__init__()
        # ignore=['model'] jest ważne, żeby checkpoint nie zapisywał całego obiektu modelu w hyperparametrach
        self.save_hyperparameters(ignore=['model']) 
        
        self.model = model
        
        #  Define Metrics
        # 'task="multiclass"' with 2 classes is standard for CrossEntropyLoss
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_prec = Precision(task="multiclass", num_classes=num_classes, average='macro')
        self.test_rec = Recall(task="multiclass", num_classes=num_classes, average='macro')
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self.test_cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, X):
        # Forward pass through the network
        return self.model(X)

    def _common_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch, batch_idx)
        
        # Logging metrics
        self.train_acc(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._common_step(batch, batch_idx)
        
        # Logging validation metrics
        self.val_acc(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
            loss, preds, y = self._common_step(batch, batch_idx)
            
            # Update all test metrics
            self.test_acc(preds, y)
            self.test_prec(preds, y)
            self.test_rec(preds, y)
            self.test_f1(preds, y)
            self.test_cm(preds, y) # Accumulate confusion matrix
            
            # Log scalar metrics
            self.log_dict({
                'test_loss': loss,
                'test_acc': self.test_acc,
                'test_precision': self.test_prec,
                'test_recall': self.test_rec,
                'test_f1': self.test_f1
            })
            return loss

    def on_test_epoch_end(self):
        """
        Executed once at the end of the test epoch.
        Calculates the final Confusion Matrix, saves it locally,
        and uploads it to W&B if the logger is active.
        """
        # 1. Compute the confusion matrix (aggregated from all batches)
        cm = self.test_cm.compute()

        # 2. Define class names (must match your dataset structure)
        classes = ['negative', 'positive']

        # 3. Save the plot locally first
        save_path = "confusion_matrix.png"

        # Ensure 'save_confusion_matrix' is imported from your utils
        save_confusion_matrix(cm, classes, save_path=save_path)

        # 4. Upload to Weights & Biases
        # We need to robustly find the WandbLogger, as 'self.logger' might be a list
        # if multiple loggers (e.g., CSV + WandB) are used.

        wandb_logger = None

        # Case A: Multiple loggers are used (self.loggers is a list)
        if self.loggers:
            for logger in self.loggers:
                if isinstance(logger, pl.loggers.WandbLogger):
                    wandb_logger = logger
                    break

        # Case B: Only one logger is used (self.logger is the object)
        elif isinstance(self.logger, pl.loggers.WandbLogger):
            wandb_logger = self.logger

        # If a WandB logger was found, log the image
        if wandb_logger:
            # Local import to avoid issues if wandb is not globally imported
            import wandb

            wandb_logger.experiment.log({
                "confusion_matrix": wandb.Image(save_path, caption="Confusion Matrix")
            })
            print(f"--> Confusion Matrix successfully uploaded to W&B!")
        else:
            print("--> WandBLogger not found. Confusion Matrix saved locally only.")

        # 6. CLEANUP: Delete the local file
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"--> Local file '{save_path}' deleted to save space.")
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        
        # Optional: Learning Rate Scheduler 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }