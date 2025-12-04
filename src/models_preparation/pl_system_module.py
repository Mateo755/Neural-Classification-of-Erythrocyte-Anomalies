import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall, ConfusionMatrix
import wandb

from src.utils.visualization import save_confusion_matrix

class MalariaClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, num_classes=2):
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
        Ta metoda uruchamia się RAZ po zakończeniu całego testowania.
        Tu obliczamy finalną macierz pomyłek.
        """
        # 1. Obliczamy macierz (zsumowaną ze wszystkich batchy)
        cm = self.test_cm.compute()
        
        # 2. Definiujemy nazwy klas (muszą pasować do folderów)
        # Możesz je też przekazać w __init__, jeśli wolisz
        classes = ['negative', 'positive'] 
        
        # 3. Rysujemy i zapisujemy lokalnie
        save_path = "confusion_matrix.png"
        save_confusion_matrix(cm, classes, save_path=save_path)
        
        # 4. Jeśli używamy WandB, wysyłamy obrazek do chmury
        # Sprawdzamy, czy logger to WandBLogger
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log({
                "confusion_matrix": wandb.Image(save_path, caption="Confusion Matrix")
            })
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        # Optional: Learning Rate Scheduler 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }