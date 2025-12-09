import torch
import torch.nn.functional as F
import pytorch_lightning as L
from torchmetrics import Accuracy

class TrainSystem(L.LightningModule):
    def __init__(self, model, learning_rate, optimizer_name="Adam", num_classes=2, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        
        # Logujemy: on_step=False (nie co batch), on_epoch=True (średnia co epokę)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

        return loss

    def configure_optimizers(self):

        if self.optimizer_name == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        elif self.optimizer_name == "RMSprop":
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        elif self.optimizer_name == "SGD":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9,
                                   weight_decay=self.weight_decay)

        elif self.optimizer_name == "Adamax":
            return torch.optim.Adamax(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        elif self.optimizer_name == "Adagrad":
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        elif self.optimizer_name == "Adadelta":
            return torch.optim.Adadelta(self.parameters(), lr=self.learning_rate,
                                        weight_decay=self.weight_decay)  # Poprawiono klasę

        elif self.optimizer_name == "Nadam":
            return torch.optim.NAdam(self.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)  # Poprawiono klasę

        else:
            raise ValueError(f"Optimizer {self.optimizer_name} not supported yet.")
    