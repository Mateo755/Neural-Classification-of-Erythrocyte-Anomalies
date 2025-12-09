import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm


# --- IMPORTS ---
# Adjust these imports to match your folder structure!
from src.models_preparation.components import BestClassifier
from src.models_preparation.pl_system_module import MalariaClassifier

class BlindTestDataset(Dataset):
    """
    Custom Dataset to handle a flat folder of images (no class subfolders).
    Returns the image tensor and the filename.
    """

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # Sort files to ensure deterministic order of predictions
        # Filter only for image extensions
        self.image_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Load image and convert to RGB (standard for ImageNet models)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Return image AND filename (filename is needed for the CSV)
        return image, img_name


def create_submission_from_checkpoint(checkpoint_path, test_dir, config, output_file="submission.csv"):
    """
    Loads a model from a .ckpt file, runs inference on a blind test set,
    and generates the submission CSV.
    """
    print(f"--> Loading checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Inference Device: {device}")

    # ---------------------------------------------------------
    # 1. RECONSTRUCT ARCHITECTURE
    # We must instantiate the model structure EXACTLY as it was trained.
    # The weights from the checkpoint will simply not fit if shapes differ.
    # ---------------------------------------------------------
    backbone = BestClassifier(
        num_classes=2,
        freeze_backbone=False,  # False is safer for loading (ensures all params exist)
        n_layers=config['n_layers'],
        hidden_dim=config['hidden_dim'],
        apply_dropout=config['apply_dropout']
        # dropout_rate doesn't affect inference, so we can skip or pass default
    )

    # ---------------------------------------------------------
    # 2. LOAD WEIGHTS INTO SYSTEM
    # We use .load_from_checkpoint() and inject our backbone object.
    # We pass dummy values for lr/optimizer because they are required by __init__
    # but strictly irrelevant for inference.
    # ---------------------------------------------------------
    model_system = MalariaClassifier.load_from_checkpoint(
        checkpoint_path,
        model=backbone,
    )

    model_system.eval()  # Switch to evaluation mode (disable Dropout, BatchNorm stats)
    model_system.freeze()  # Disable gradient calculation
    model_system.to(device)

    # ---------------------------------------------------------
    # 3. PREPARE DATA
    # Use ImageNet normalization and Resize from config
    # ---------------------------------------------------------
    test_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = BlindTestDataset(img_dir=test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )

    # ---------------------------------------------------------
    # 4. PREDICTION LOOP
    # ---------------------------------------------------------
    filenames = []
    predictions = []

    print(f"--> Starting prediction on {len(test_dataset)} images...")

    for imgs, fnames in tqdm(test_loader, desc="Inference"):
        imgs = imgs.to(device)

        with torch.no_grad():
            logits = model_system(imgs)
            # Apply Softmax to get probabilities (optional, good for debugging)
            probs = torch.softmax(logits, dim=1)
            # Get the class index (0 or 1) with higher probability
            preds = torch.argmax(probs, dim=1)

        predictions.extend(preds.cpu().numpy())
        filenames.extend(fnames)

    # ---------------------------------------------------------
    # 5. SAVE CSV
    # ---------------------------------------------------------
    df = pd.DataFrame({
        "filename": filenames,
        "prediction": predictions
    })

    # Optional: Verify mapping
    # Usually: 0 -> Negative, 1 -> Positive (based on alphabetical folder order during training)
    print("--> Sample predictions:")
    print(df.head())

    df.to_csv(output_file, index=False)
    print(f"Submission saved to: {output_file}")


# --- EXECUTION BLOCK ---
if __name__ == "__main__":

    # A. CONFIGURATION (Must match the training config!)
    # Ideally, copy this from your training script or WandB config
    INFERENCE_CONFIG = {
        "n_layers": 4,  # EXAMPLE
        "hidden_dim": 128,  # EXAMPLE
        "apply_dropout": False,  # EXAMPLE
        "img_size": 224,
        "batch_size": 32
    }

    # B. PATHS
    # Replace with your actual best checkpoint path
    CKPT_PATH = "checkpoints/malaria-epoch=07-val_loss=0.07.ckpt"
    # Path to the folder containing flat images
    TEST_IMAGES_DIR = "./malaria_dataset/test"

    # Check if files exist
    if os.path.exists(CKPT_PATH) and os.path.exists(TEST_IMAGES_DIR):
        create_submission_from_checkpoint(
            checkpoint_path=CKPT_PATH,
            test_dir=TEST_IMAGES_DIR,
            config=INFERENCE_CONFIG
        )
    else:
        print("Error: Checkpoint or Test Directory not found.")