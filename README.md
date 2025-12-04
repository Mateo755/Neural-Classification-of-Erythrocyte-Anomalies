# Project: Neural Classification of Erythrocyte Anomalies

## 1. Project Overview
In low-resource hematology settings, manual screening of blood smears for intracellular parasites is time-consuming and error-prone. This project aims to automate the triage process by developing a Deep Learning model capable of distinguishing between healthy erythrocytes (red blood cells) and those containing a specific **intracellular pathogen**.

**Goal:** Design, train, and validate a **Convolutional Neural Network (CNN)** to perform binary classification on single-cell images.

## 2. The Dataset
**Download Link:** [Dataset](https://chmura.put.poznan.pl/s/azZNaEokFpaNfyt)

The proprietary dataset consists of segmented “patches” (Regions of Interest) extracted from thin blood smear slides stained with Giemsa. Each image contains a single cell.

### Data Structure
The data is split into two sets:
* **`train`** (~22,000 labeled images):
    * **`negative`**: Healthy/Control samples.
    * **`positive`**: Infected/Anomalous samples.
* **`test`** (~5,500 unlabeled images):
    * No ground truth labels provided.
    * Used for generating final predictions for grading.

**Note:** Images have varying resolutions and aspect ratios. A robust pre-processing strategy is required.

## 3. Technical Objectives

### A. Data Pre-processing & Augmentation
Implement a pipeline to:
1.  **Resize/Rescale** images to a fixed input size (e.g., 64x64, 128x128, or 224x224).
2.  **Normalize** pixel intensity values.
3.  Implement **Data Augmentation** on the training set (rotations, flips, brightness) to prevent overfitting.

### B. Neural Network Architecture
Construct a CNN using one of the following paths:
* **Custom Architecture:** Design your own stack of Convolutional, Max-Pooling, and Dense layers. Justify kernel sizes and depth.
* **Transfer Learning:** Use a pre-trained backbone (e.g., VGG-16, ResNet-18, MobileNet) with a custom classification head. Explain freezing/unfreezing strategy.

### C. Training Loop
* **Loss Function:** Select a loss function appropriate for binary classification.
* **Optimizer:** Use an adaptive optimizer or SGD with momentum.
* **Validation:** Create an internal validation split (e.g., 80/20) from the `train` set to monitor loss and prevent overfitting.

## 4. Deliverables

### Part 1: Short Report (PDF)
Must include:
* Neural network architecture, loss function, optimizer, and hyperparameters.
* Model interpretability visualizations (e.g., Grad-CAM via Captum).

### Part 2: The “Blind” Test Submission
Run the trained model on the **`test`** folder images and generate a CSV file.

* **Filename:** `submission.csv`
* **Format:** Header with two columns: `filename`, `prediction`
* **Values:** `0` or `1`
* **Requirement:** Filenames must match exactly.