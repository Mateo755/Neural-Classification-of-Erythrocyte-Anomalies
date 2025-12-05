import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
import torch

# Import your project modules
from src.data_preparation.data_module import MalariaDataModule
from builder import ModelBuilder
from system import TrainSystem

def objective_frozen(trial: optuna.trial.Trial):
    """
    Optuna objective function for hyperparameter optimization with a frozen backbone.
    """

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
    
    trainer = pl.Trainer(
        max_epochs=6,                   # Short epochs for search
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

    # Dowlad final metrics
    # We use get with default to avoid KeyError if metric is missing
    final_val_acc = trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()
    final_train_acc = trainer.callback_metrics.get("train_acc", torch.tensor(0.0)).item()
    final_train_loss = trainer.callback_metrics.get("train_loss", torch.tensor(0.0)).item()
    
    # --- SAVE TO OPTUNA DATABASE ---
    trial.set_user_attr("train_acc", final_train_acc)
    trial.set_user_attr("train_loss", final_train_loss)

    gap = final_train_acc - final_val_acc
    trial.set_user_attr("overfit_gap", gap)
    
    # Return the best validation accuracy achieved
    return final_val_acc



def objective_finetune(trial: optuna.trial.Trial):
    """
    Stage 2: Search for fine-grained Learning Rate and Weight Decay.
    Backbone is UNFROZEN. Architecture is fixed based on Stage 1 results.
    """
    
    # === WINNER CONFIG FROM STAGE 1 (Update this manually after Stage 1!) ===
    FIXED_BASE_MODEL = "resnet18"  # Example
    FIXED_LAYERS = 1               # Example
    FIXED_HIDDEN = 256             # Example
    FIXED_BATCH_SIZE = 32          # Example
    FIXED_OPTIMIZER = "Adam"       # Example
    # ========================================================================

    # 1. Hyperparameter Suggestions (Focus on LR and Regularization)
    # Note: LR range is much lower now (e.g., 1e-6 to 1e-4)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # 2. Build Components
    backbone = ModelBuilder(
        base_model_name=FIXED_BASE_MODEL,
        num_classes=2,
        freeze_backbone=False,      # <--- UNFROZEN (The key change)
        num_hidden_layers=FIXED_LAYERS,
        hidden_dim=FIXED_HIDDEN,
        # We can keep dropout fixed or search it again if we want
        use_dropout=True, dropout_rate=0.4 
    )
    
    model = TrainSystem(
        model=backbone,
        learning_rate=learning_rate,
        optimizer_name=FIXED_OPTIMIZER,
        # weight_decay=weight_decay  <-- Uncomment if you added this to System
    )
    
    dm = MalariaDataModule(data_dir='./malaria_dataset', batch_size=FIXED_BATCH_SIZE)
    
    # 3. Train
    logger = CSVLogger("optuna_logs_finetune", name=f"trial_{trial.number}")
    
    trainer = pl.Trainer(
        max_epochs=3, 
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        logger=logger,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")]
    )
    
    trainer.fit(model, datamodule=dm)
    
    # 4. Return
    final_val_acc = trainer.callback_metrics.get("val_acc", torch.tensor(0.0)).item()
    
    trial.set_user_attr("train_acc", trainer.callback_metrics.get("train_acc", 0.0).item())
    trial.set_user_attr("train_loss", trainer.callback_metrics.get("train_loss", 0.0).item())
    
    return final_val_acc