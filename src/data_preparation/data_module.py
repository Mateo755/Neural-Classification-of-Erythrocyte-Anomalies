import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import pytorch_lightning as pl

class MalariaDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./malaria_dataset", batch_size: int = 32, img_size: int = 224):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size

        # --- CORRECT IMAGENET STATISTICS ---
        # These are crucial for pre-trained models to work correctly!
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        
        # Path to the training folder (contains 'positive' and 'negative' subfolders)
        self.train_dir = os.path.join(data_dir, 'train')

        # 1. Define Transforms
        
        # Training Transforms (With Augmentation)
        # We use this to artificially increase dataset diversity and prevent overfitting.
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(20),       # Rotate +/- 20 degrees
            transforms.RandomHorizontalFlip(),   # Mirror image
            transforms.RandomVerticalFlip(),     # Flip upside down (cells have no orientation)
            transforms.ColorJitter(brightness=0.1, contrast=0.1), # Simulate lighting variations
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        # Evaluation Transforms (Clean)
        # Used for Validation and Test sets. No random changes, just resizing and normalizing.
        self.eval_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def setup(self, stage=None):
        """
        Splits the data into Train/Val/Test and applies appropriate transformations.
        Approach: We use two separate ImageFolder objects (one augmented, one clean)
        and map indices to them using Subsets.
        """
        if stage == "fit" or stage == "test" or stage is None:
            # STEP 1: Load directory structure to get the total number of images
            # We use a dummy dataset just to read filenames and length
            dummy_ds = datasets.ImageFolder(self.train_dir)
            n_total = len(dummy_ds)
            
            # STEP 2: Calculate split sizes (80% Train, 10% Val, 10% Test)
            train_len = int(0.8 * n_total)
            val_len = int(0.1 * n_total)
            # The rest goes to test
            
            # STEP 3: Generate random indices for the split
            # We use a fixed seed generator for reproducibility
            g = torch.Generator().manual_seed(42)
            indices = torch.randperm(n_total, generator=g).tolist()
            
            train_idx = indices[:train_len]
            val_idx = indices[train_len : train_len + val_len]
            test_idx = indices[train_len + val_len :]

            # STEP 4: Create TWO separate dataset objects with different transforms
            # Dataset A: With Augmentation (for Training)
            train_dataset_obj = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
            # Dataset B: Without Augmentation (for Validation/Testing)
            eval_dataset_obj = datasets.ImageFolder(self.train_dir, transform=self.eval_transform)

            # STEP 5: Create Subsets by mapping indices to the correct dataset object
            self.train_ds = Subset(train_dataset_obj, train_idx)
            self.val_ds = Subset(eval_dataset_obj, val_idx)
            self.test_ds = Subset(eval_dataset_obj, test_idx)

            print(f"--- Data Module Setup Complete ---")
            print(f"Train samples: {len(self.train_ds)} (Augmented)")
            print(f"Val samples:   {len(self.val_ds)} (Clean)")
            print(f"Test samples:  {len(self.test_ds)} (Clean)")
            print(f"Classes:       {dummy_ds.classes}") # Should be ['negative', 'positive']

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)#, num_workers=2, persistent_workers=True,  pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)#, num_workers=2, persistent_workers=True,  pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)#, num_workers=2, persistent_workers=True,  pin_memory=True)