import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

def compute_features(network, dataloader, compute_device, num_classes):
    """
    Computes all logits for a validation set
    """
    # Arrays for storing the logits per instance and ground truth per instance
    features = np.zeros((len(dataloader.dataset), num_classes), dtype=float)
    ground_truth = np.zeros((len(dataloader.dataset)))

    for index, sample in enumerate(dataloader):
        # Load in batch image data
        instance_data = sample[0]
        instance_data.requires_grad = False

        # Load in batch label data
        label_data = sample[1].item()

        # Send image and label data to device
        instance_data = instance_data.to(compute_device)

        # Forward pass and get the output predictions
        logits = network(instance_data).squeeze()

        # Save to arrays
        features[index, :] = logits.detach().cpu().numpy()
        ground_truth[index] = label_data

        # Get the softmaxes
        softmaxes = F.softmax(logits, dim=0).detach().cpu().numpy()

    return features, ground_truth

       
def platt_scaling(features, ground_truth, num_classes):
    """
    Performs Platt Scaling calibration
    """
    calibrated_models = {}
    for num in range(num_classes):
        logistic_regression_targets = (ground_truth == num).astype(float)
        calibrated_models[num] = train_logistic_regression_model(features[:, num], logistic_regression_targets)
    return calibrated_models

def train_logistic_regression_model(features, targets):
    """
    Trains a logistic regression model for Platt Scaling
    """
    # If the class doesn't exist or the targets is all one class
    if np.sum(targets) == 0 or np.sum(targets) == len(targets):
        return np.sum(targets)
    else:
        features = features.reshape((-1, 1))
        model = LogisticRegression(solver='lbfgs')
        model.fit(features, targets)
        return model

