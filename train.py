import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import wandb
import torch

# Importy z naszych nowych modułów
from src.data_preparation import MalariaDataModule
from src.models_preparation.components import CustomResnet
from src.models_preparation import MalariaClassifier
from src.utils.visualization import plot_training_metrics

def main():

    USE_WANDB = True
    exp_name = "ResNet50-Refactored"

    # --- 0. CONFIGURATION ---
    HYPERPARAMETERS = {
        # Architecture Params (from model_builder)
        "n_layers": 5,
        "hidden_dim": 128,
        "apply_dropout": False,
        "dropout_rate": 0.2,
        "freeze_backbone": False,
        "base_model": "resnet50",
        "base_model_weights": "IMAGENET1K_V2",

        # Training Params
        "batch_size": 32,
        "img_size": 224,
        "epochs": 1,
        "learning_rate": 9.784011201151404e-05,
        "optimizer": "Adam",
        "weight_decay": 2.3314715675645093e-06,
        "early_stopping_patience": 5,
        "seed": 42
    }


    pl.seed_everything(HYPERPARAMETERS['seed'])

    # 1. Dane
    dm = MalariaDataModule(data_dir='./malaria_dataset', batch_size=HYPERPARAMETERS['batch_size'], img_size=HYPERPARAMETERS['img_size'])

    # 2. Model
    backbone = CustomResnet(num_classes=2,
                            freeze_backbone=HYPERPARAMETERS['freeze_backbone'],
                            n_layers=HYPERPARAMETERS['n_layers'],
                            hidden_dim=HYPERPARAMETERS['hidden_dim'],
                            apply_dropout=HYPERPARAMETERS['apply_dropout'],
                            dropout_rate=HYPERPARAMETERS['dropout_rate']
                            )

    model = MalariaClassifier(model=backbone,
                              learning_rate=HYPERPARAMETERS['learning_rate'],
                              weight_decay = HYPERPARAMETERS['weight_decay'])

    # 3. Loggers
    csv_logger = CSVLogger("logs", name="malaria_resnet50")
    loggers = [csv_logger]
    
    if USE_WANDB:
        wandb.login()
        wandb_logger = WandbLogger(project="Malaria-Classification",
                                   name=exp_name,
                                   config=HYPERPARAMETERS)

        loggers.append(wandb_logger)

    # 4. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='malaria-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss', mode='min', save_top_k=1
    )
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=HYPERPARAMETERS['early_stopping_patience'],
                                        mode='min')

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=HYPERPARAMETERS['epochs'],
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