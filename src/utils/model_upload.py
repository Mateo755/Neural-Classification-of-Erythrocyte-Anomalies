import wandb
import os

def manual_artifact_upload(trainer, callback, artifact_name="my-best-model"):
    """
    Manually uploads the best model checkpoint as a W&B Artifact.
    Allows adding custom metadata and additional files (like submission.csv).
    """
    print("\n--- Starting Manual Artifact Upload ---")
    
    # 1. Retrieve the path to the best model saved locally
    best_model_path = callback.best_model_path
    
    if not best_model_path:
        print("Error: No checkpoint found! Did training fail?")
        return

    print(f"Found best model locally at: {best_model_path}")

    # 2. Create the Artifact object
    # type="model" is crucial for W&B to recognize it as a neural network
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description="Manually logged ResNet18 model (best val_loss)",
        metadata={
            "val_loss": callback.best_model_score.item(), # Log the best score
            "architecture": "ResNet18",
            "dataset": "Malaria Project 1",
            "img_size": 128
        }
    )

    # 3. Add the model file to the artifact
    artifact.add_file(best_model_path)
    
    # Optional: Add other files to the same package (e.g., your predictions)
    # if os.path.exists("submission.csv"):
    #     artifact.add_file("submission.csv") 

    # 4. Upload to the cloud
    # We access the W&B run object via the logger
    if isinstance(trainer.logger, L.loggers.WandbLogger):
        print("Uploading artifact to W&B... (this may take a moment)")
        trainer.logger.experiment.log_artifact(artifact)
        print("Artifact uploaded successfully!")
        
        # Optional: Wait for upload to finish to ensure data integrity before closing
        artifact.wait() 
    else:
        print("Warning: WandbLogger is not active. Skipping upload.")




# --- USAGE AT THE END OF SCRIPT ---

# Execute this after trainer.test()
# if USE_WANDB:
#     # You can name the artifact whatever you want, e.g., 'resnet-final-v1'
#     manual_artifact_upload(trainer, checkpoint_callback, artifact_name="resnet-final-v1")