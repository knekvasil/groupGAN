# utils/diagnostics.py

"""
This module provides functions for evaluating and visualizing the performance of
binary classification models, particularly designed for models outputting a single
probability score (e.g., after a Sigmoid activation).

It includes functions for:
- Running inference on a dataset and collecting true labels and predicted probabilities.
- Calculating key classification metrics (accuracy, precision, recall, F1-score).
- Generating and plotting a confusion matrix.
- Plotting the Receiver Operating Characteristic (ROC) curve and calculating AUC.
- Visualizing the distribution of true class labels in a dataset.

The functions are designed to work with PyTorch models and dataloaders,
and utilize popular libraries like numpy, sklearn, matplotlib, and seaborn
for numerical operations, metric calculation, and plotting.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,  # Not used in plots below, but good to keep
)
from typing import Tuple, Dict
import pandas as pd

plt.style.use("ggplot")


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Core evaluation function for binary classifiers with a single Sigmoid output.

    Runs inference on the provided dataloader, collects true labels and model
    outputs (probabilities), calculates standard classification metrics, and
    returns the metrics along with the collected labels and probabilities.

    Args:
        model: The PyTorch binary classification model to evaluate.
               Assumes the model outputs a single value per sample, typically
               interpreted as the probability of the positive class (after Sigmoid).
        dataloader: A PyTorch DataLoader providing the data to evaluate on.
                    It should yield batches of (inputs, labels).
        device: The device to run the evaluation on ('cuda' or 'cpu'). Defaults
                to 'cuda' if available, otherwise 'cpu'.

    Returns:
        - metrics_dict: Dictionary of performance metrics including accuracy,
                        precision, recall, F1-score for both classes and weighted
                        averages, and the full classification report dictionary.
        - all_labels: Numpy array of true labels from the dataloader (shape N,).
        - all_probs: Numpy array of predicted probabilities for the positive
                     class (shape N, 1).
    """
    model.eval()
    all_labels = []
    all_outputs = []  # Store raw outputs from the model (after Sigmoid)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Outputs are already 0-1 after Sigmoid
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_outputs = torch.cat(all_outputs)  # Shape (N, 1)
    all_labels = torch.cat(all_labels)  # Shape (N, 1)

    # Use the model output directly as probabilities for the positive class
    all_probs = all_outputs.numpy()  # Shape (N, 1)

    # Get predictions by thresholding the probabilities (e.g., at 0.5)
    # Squeeze to get shape (N,) for thresholding comparison
    preds = (all_probs.squeeze(1) > 0.5).astype(int)  # Shape (N,)

    # Sklearn metrics expect labels and predictions of shape (N,) or (N, 1)
    # Let's pass them as (N,) by squeezing the labels too for consistency with preds
    report = classification_report(
        all_labels.squeeze(1).numpy(), preds, output_dict=True, zero_division=0
    )

    # You can adjust zero_division if needed, 0 means metrics for classes with no predicted samples will be 0
    # report = classification_report(all_labels.numpy(), preds, output_dict=True, zero_division=1) # Use 1 if you want warning or different behavior

    metrics = {
        "accuracy": report["accuracy"],
        "precision_0": report["0.0"]["precision"] if "0.0" in report else 0.0,
        "recall_0": report["0.0"]["recall"] if "0.0" in report else 0.0,
        "f1_0": report["0.0"]["f1-score"] if "0.0" in report else 0.0,
        "precision_1": report["1.0"]["precision"] if "1.0" in report else 0.0,
        "recall_1": report["1.0"]["recall"] if "1.0" in report else 0.0,
        "f1_1": report["1.0"]["f1-score"] if "1.0" in report else 0.0,
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "class_report": report,
    }

    return (
        metrics,
        all_labels.squeeze(1).numpy(),
        all_probs,
    )  # Return labels and probs as shape (N,) and (N,1) respectively


# --- Plotting Functions - Adjust to handle (N, 1) probs and (N,) labels ---


def plot_confusion_matrix(
    labels: np.ndarray, probs: np.ndarray, class_names: list = ["Fake", "Real"]
):
    """
    Generate and display a confusion matrix based on true labels and predicted probabilities.

    Predictions are made by thresholding the probabilities at 0.5.

    Args:
        labels: Numpy array of true labels (shape N, or N, 1).
        probs: Numpy array of predicted probabilities for the positive class
               (shape N, 1).
        class_names: A list of strings for the class names (e.g., ['Negative', 'Positive']).
                     Defaults to ["Fake", "Real"].
    """
    # Labels should be (N,), probs (N, 1)
    preds = (probs.squeeze(1) > 0.5).astype(int)  # Correct preds from probabilities

    # Ensure labels is (N,) if it came as (N, 1)
    if labels.ndim > 1 and labels.shape[1] == 1:
        labels = labels.squeeze(1)

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def plot_roc_curve(labels: np.ndarray, probs: np.ndarray):
    """
    Plot ROC curve for binary classification based on true labels and predicted probabilities.

    Calculates and displays the Area Under the Curve (AUC). Handles cases where
    only one class is present in the labels.

    Args:
        labels: Numpy array of true labels (shape N, or N, 1).
        probs: Numpy array of predicted probabilities for the positive class
               (shape N, 1).
    """
    # Labels should be (N,), probs (N, 1)
    # roc_curve expects probabilities of the positive class
    # If probs is (N, 1), sklearn uses the single column
    # If labels is (N, 1), sklearn handles it

    # Ensure labels is (N,) if it came as (N, 1)
    if labels.ndim > 1 and labels.shape[1] == 1:
        labels = labels.squeeze(1)

    # Check if there are at least two unique classes in the true labels
    if len(np.unique(labels)) < 2:
        print("Cannot plot ROC curve: Only one class present in true labels.")
        roc_auc = 0.0  # AUC is not meaningful
        # Optional: you could return early here if you don't want to show an empty plot
        return
    else:
        fpr, tpr, _ = roc_curve(labels, probs)  # Use probabilities for class 1
        roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


def plot_class_distribution(labels: np.ndarray):
    """
    Plot the distribution of true class labels in a dataset.

    Args:
        labels: Numpy array of true labels (shape N, or N, 1). Expected to contain
                integer labels (e.g., 0 and 1).
    """
    # Ensure labels is (N,) if it came as (N, 1)
    if labels.ndim > 1 and labels.shape[1] == 1:
        labels = labels.squeeze(1)

    plt.figure(figsize=(6, 4))
    # Use pd.Series(labels).value_counts().sort_index() to ensure 0 and 1 are in order
    # Use .reset_index() and .rename() for seaborn compatibility if needed,
    # but direct pandas plot works fine here.
    pd.Series(labels).value_counts().sort_index().plot(kind="bar")
    plt.xticks([0, 1], ["Fake", "Real"], rotation=0)
    plt.title("True Class Distribution")  # Clarified title
    plt.ylabel("Count")
    plt.show()
