import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import wandb
import torch

# Importy z naszych nowych modułów
from src.data_preparation import MalariaDataModule
from src.models_preparation import Resnet
from src.models_preparation import MalariaClassifier
from src.utils.visualization import plot_training_metrics

def main():
    # 0. Konfiguracja
    pl.seed_everything(42)
    USE_WANDB = True
    BATCH_SIZE = 8
    IMG_SIZE = 224
    LR = 1e-3
    EPOCHS = 10

    # 1. Dane
    dm = MalariaDataModule(data_dir='./malaria_dataset', batch_size=BATCH_SIZE, img_size=IMG_SIZE)

    # 2. Model
    backbone = Resnet(num_classes=2, freeze_backbone=True)
    model = MalariaClassifier(model=backbone, learning_rate=LR)

    # 3. Loggers
    csv_logger = CSVLogger("logs", name="malaria_resnet")
    loggers = [csv_logger]
    
    if USE_WANDB:
        wandb.login()
        wandb_logger = WandbLogger(project="Malaria-Classification", name="ResNet18-Refactored")
        loggers.append(wandb_logger)

    # 4. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='malaria-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss', mode='min', save_top_k=1
    )
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=loggers,
        log_every_n_steps=10
    )

    # 6. Trening
    trainer.fit(model, datamodule=dm)

    # 7. Test
    trainer.test(model, datamodule=dm, ckpt_path='best')

    # 8. Wykresy (po treningu)
    # Zwróć uwagę, że ścieżka do logów może się różnić w zależności od wersji loggera
    print(f"Plotting metrics from: {csv_logger.log_dir}")
    plot_training_metrics(csv_logger.log_dir)

    if USE_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()