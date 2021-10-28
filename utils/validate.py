import sys

import torch
import torch.nn.functional as F

import numpy as np

from utils.recorder import EvaluateRecorder

def validation(network, dataloader, compute_device, experiment, results_directory, classification_loss_func):
    # Get the batch size
    batch_size = dataloader.batch_sampler.batch_size

    # Instantiate two arrays to keep track of the ground truth label and network prediction for each instance
    num_instances = len(dataloader.dataset)
    true_classes = np.zeros(num_instances)
    predicted_classes = np.zeros(num_instances)

    network.eval()

    val_loss = 0.0

    with torch.no_grad():
        # Begin evaluating the neural network
        for batch_num, batch_sample in enumerate(dataloader):
            # Load in batch image data
            image_data = batch_sample[0]
            image_data.requires_grad = False

            # Load in batch label data
            label_data = batch_sample[1]
            label_data.requires_grad = False

            # Send image and label data to device
            image_data = image_data.to(compute_device)
            label_data = label_data.to(compute_device)

            # Forward pass and get the output predictions
            predictions = network(image_data)

            # Accumulate the validation loss for each batch
            loss = classification_loss_func(predictions, label_data)
            val_loss += loss.item()

            # Get the flat prediction
            predictions = torch.argmax(predictions, dim=1)

            # Record the actual and predicted labels for the instance
            true_classes[ batch_num * batch_size : min( (batch_num + 1) * batch_size, num_instances) ] = label_data.detach().cpu().numpy()
            predicted_classes[ batch_num * batch_size : min( (batch_num + 1) * batch_size, num_instances) ] = predictions.detach().cpu().numpy() 

    recorder = EvaluateRecorder(results_directory, experiment, 'val')
    accuracy = recorder.record(true_classes, predicted_classes, dataloader.dataset.classes, save=False)
    
    return (val_loss / (batch_num + 1)), accuracy