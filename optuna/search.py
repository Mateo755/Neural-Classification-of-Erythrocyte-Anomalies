import optuna
import torch

from objectives import objective_frozen, objective_finetune

# ==============================================================================
# RUN OPTIMIZATION
# ==============================================================================

# Define study name (Change this when switching strategies so you don't mix DBs)
STUDY_NAME = "Malaria_Stage1_Frozen" 
# STUDY_NAME = "Malaria_Stage2_Finetune"



if __name__ == "__main__":
    # Ensure reproducibility (Optional but good for debugging)
    torch.manual_seed(42)

    print(f"--- Starting Optuna Search: {STUDY_NAME} ---")

    # Create Study
    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_NAME}.db", # Creates a separate DB file for each stage
        load_if_exists=True
    )

    # Optimize
    # n_trials depends on your time budget. 
    # Frozen: 20-30 trials. Finetune: 10-15 trials.
    
    # Option A: Fixed number of trials (Best for guarantee of quality)
    study.optimize(objective_frozen, n_trials=30)
    #study.optimize(objective_finetune, n_trials=15)
    
    # Option B: Timebox (Best for "run overnight")
    # study.optimize(objective_frozen, n_trials=100, timeout=8 * 3600)
    
    print("\n--- Optimization Finished ---")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Val Acc: {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")