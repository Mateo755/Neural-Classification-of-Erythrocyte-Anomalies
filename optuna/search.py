import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as L

from src.data_preparation.data_module import MalariaDataModule
from optuna.builder import ModelBuilder
from optuna.system import TrainSystem

def objective(trial: optuna.trial.Trial):
    # -------------------------------------------------
    # 1. Hyperparameter Suggestions 
    # -------------------------------------------------
    
    # Architecture choice
    base_model_str = trial.suggest_categorical(
        "base_model", ["resnet18", "resnet50", "mobilenet_v2", "vgg16"]
    )

    # Architecture - Head (
    # How many dense layers 
    n_layers = trial.suggest_int("n_layers", 1, 4)
    # How wide should these layers be
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    
    # Regularization
    dropout_rate = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3, 0.4, 0.5])
    apply_dropout = trial.suggest_categorical("apply_dropout", [True, False])
    
    # Training Config
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    optimizer_str = trial.suggest_categorical(
        "optimizer", ["Adam", "RMSprop", "SGD", "Adagrad", "Adadelta", "Adamax", "Nadam"]
    )
    
    # -------------------------------------------------
    # 2. Setup Components
    # -------------------------------------------------
    
    # Build Model Architecture (nn.Module)
    backbone = ModelBuilder(
        base_model_name=base_model_str,
        num_classes=2,
        dropout_rate=dropout_rate,
        use_dropout=apply_dropout,
        freeze_backbone=True,
        num_hidden_layers=n_layers, 
        hidden_dim=hidden_dim
    )
    
    # Build Lightning System
    model = TrainSystem(
        model=backbone,
        learning_rate=learning_rate,
        optimizer_name=optimizer_str
    )
    
    # Build DataModule
    # Note: We re-initialize it to change batch_size
    dm = MalariaDataModule(data_dir='./malaria_dataset', batch_size=batch_size)
    
    # -------------------------------------------------
    # 3. Training
    # -------------------------------------------------
    
    # Use CSVLogger to keep W&B clean during search
    logger = CSVLogger("optuna_logs", name=f"trial_{trial.number}")
    
    trainer = L.Trainer(
        max_epochs=5,                   # Short epochs for search
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,     # Save disk space
        logger=logger,
        callbacks=[
            # This corresponds to TF's EarlyStopping/Pruning
            PyTorchLightningPruningCallback(trial, monitor="val_acc")
        ]
    )
    
    # Fit
    trainer.fit(model, datamodule=dm)
    
    # -------------------------------------------------
    # 4. Return Metric
    # -------------------------------------------------
    
    # Return the best validation accuracy achieved
    if "val_acc" in trainer.callback_metrics:
        return trainer.callback_metrics["val_acc"].item()
    else:
        return 0.0

# ==============================================================================
# RUN OPTIMIZATION
# ==============================================================================
if __name__ == "__main__":
    # Create Study
    study = optuna.create_study(
        direction="maximize",
        study_name="Malaria_Architecture_Search",
        storage="sqlite:///malaria_hpo.db", # Save progress to DB file
        load_if_exists=True
    )
    
    print("Starting Optuna Search...")
    study.optimize(objective, n_trials=15, timeout=7200) # 2 hours timeout

    print("\n--- Best Trial ---")
    print(study.best_params)