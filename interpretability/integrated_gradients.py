import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import sys

# --- CAPTUM IMPORTS ---
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# Import your project modules
# 1.Download current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2.Go up one level to the parent directory (the root directory where src is located)
parent_dir = os.path.dirname(current_dir)

# 3, Add this parent directory to the Python path
sys.path.append(parent_dir)

# --- PROJECT IMPORTS ---
# Using the same imports as your evaluate.py to ensure compatibility
from src.models_preparation.components import BestClassifier
from src.models_preparation.pl_system_module import MalariaClassifier

# --- CONFIGURATION ---
# These specific mean/std values are from ImageNet (standard for ResNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CONFIG = {
    "n_layers": 4,
    "hidden_dim": 128,
    "apply_dropout": False,
    "img_size": 224
}


def load_model(checkpoint_path):
    """
    Loads the trained model from the checkpoint.
    """
    print(f"--> Loading model from: {checkpoint_path}")

    # Reconstruct the architecture structure
    backbone = BestClassifier(
        num_classes=2,
        freeze_backbone=False,
        n_layers=CONFIG['n_layers'],
        hidden_dim=CONFIG['hidden_dim'],
        apply_dropout=CONFIG['apply_dropout']
    )

    # Load weights
    model_system = MalariaClassifier.load_from_checkpoint(
        checkpoint_path,
        model=backbone,
    )

    model_system.eval()
    model_system.freeze()
    return model_system


def preprocess_image(image_path):
    """
    Loads an image and applies the necessary transforms for the model.
    Returns:
        input_tensor: (1, C, H, W) tensor for the model/Captum
        original_image: PIL image for visualization
    """
    transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image)

    # Add batch dimension [1, 3, 224, 224]
    input_tensor = input_tensor.unsqueeze(0)

    return input_tensor, image


def denormalize(tensor):
    """
    Reverses the ImageNet normalization to display the image correctly in the plot.
    """
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(MEAN, STD)],
        std=[1 / s for s in STD]
    )
    return inv_normalize(tensor)


def interpret_prediction(model, image_path, output_plot_name="interpretation_result.png"):
    """
    Uses Integrated Gradients to interpret the model's decision.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1. Prepare Data
    input_tensor, original_pil = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    # 2. Get Model Prediction first
    # We need to know which class the model predicts to explain THAT specific class
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        prediction_score, pred_label_idx = torch.max(probs, dim=1)
        pred_label_idx = pred_label_idx.item()

    class_names = ["Negative", "Positive"]  # Assuming 0: Negative, 1: Positive
    print(f"--> Prediction: {class_names[pred_label_idx]} ({prediction_score.item():.4f})")

    # 3. Apply Integrated Gradients
    # Define the algorithm
    ig = IntegratedGradients(model)

    # Calculate attributions
    # target=pred_label_idx ensures we explain why it chose the predicted class
    print("--> Calculating Integrated Gradients...")
    attributions, delta = ig.attribute(
        input_tensor,
        target=pred_label_idx,
        n_steps=50,  # More steps = more accurate approx, but slower
        return_convergence_delta=True
    )
    print(f"--> Convergence Delta: {delta.item()}")

    # 4. Prepare for Visualization
    # Transpose to (H, W, C) for Matplotlib
    # We use a denormalized version of the input to show the actual colors
    denorm_img = denormalize(input_tensor.squeeze(0).cpu()).permute(1, 2, 0).numpy()

    # Transpose attributions to (H, W, C)
    attributions_np = attributions.squeeze(0).cpu().detach().numpy()
    attributions_np = np.transpose(attributions_np, (1, 2, 0))

    # 5. Visualize
    print(f"--> Generating visualization: {output_plot_name}")

    fig, _ = viz.visualize_image_attr_multiple(
        attributions_np,
        denorm_img,
        ["original_image", "blended_heat_map", "heat_map"],
        ["all", "absolute_value", "positive"],
        show_colorbar=True,
        titles=[
            f"Original\n({class_names[pred_label_idx]})",
            "Overlay (Integrated Gradients)",
            "Heatmap Only"
        ],
        fig_size=(14, 6)
    )


    # Define the output folder name
    output_folder = os.path.join(current_dir, "interpretability_outputs")

    # Create the directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Combine folder and filename
    save_path = os.path.join(output_folder, output_plot_name)

    plt.savefig(save_path)
    plt.close()
    print("--> Done.")


# --- MAIN EXECUTION ---
if __name__ == "__main__":

    # Path to your trained checkpoint
    CKPT_PATH = os.path.join(parent_dir, "checkpoints", "malaria-epoch=07-val_loss=0.07.ckpt")

    SAMPLE_IMAGE = os.path.join(parent_dir, "malaria_dataset", "train", "negative",
                                "0a17501b043866debe8abab505b5eab5.png")

    # Path to a specific image you want to test
    # Change this to a real file path in your dataset!
    #SAMPLE_IMAGE = "./malaria_dataset/train/positive/0af0c0a77b561f381471126bdb0876e5.png"
    #SAMPLE_IMAGE = "./malaria_dataset/train/negative/0a17501b043866debe8abab505b5eab5.png"

    if os.path.exists(CKPT_PATH) and os.path.exists(SAMPLE_IMAGE):
        # Load System
        model_system = load_model(CKPT_PATH)

        # Run Interpretation
        interpret_prediction(model_system, SAMPLE_IMAGE)
    else:
        print("Error: Checkpoint or Image file not found.")
        print(f"Checked: {CKPT_PATH}")
        print(f"Checked: {SAMPLE_IMAGE}")