import math

## Deep learning and array processing libraries
import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F 

## Inner-Project Imports
from davis_utils.tree import update_softmax
from davis_utils.tree import lowest_common_subsumer

def process_instance(confidence_threshold, tree, network, compute_device, sample, classes, nbins):
    """
    Send one instance through the inference procedure
    """
    # Load in batch image data
    image_data = sample[0]
    image_data.requires_grad = False
    image_data = image_data.to(compute_device)

    # Load in batch label data
    ground_truth_index = sample[1].item()

    # Forward pass and get the output predictions
    logits = network(image_data).squeeze()

    # Get the softmax vector
    softmaxes = F.softmax(logits, dim=0).detach().cpu().numpy()

    # Get the classifiers predictions
    predicted_index = np.argmax(softmaxes)

    # Update the softmaxes on the tree
    update_softmax(tree.unknown, softmaxes)

    # Get the softmax value for the flat prediction
    s = softmaxes[predicted_index] 

    # Quantize the neural network softmax value into a histogram bin
    sq = quantize_softmax(s, nbins=nbins)

    # Get label of neural network flat prediction
    predicted_label = classes[predicted_index]
    ground_truth_label = classes[ground_truth_index]

    # Get the TreeNode of that label
    node = getattr(tree, predicted_label)
    predicted = getattr(tree, predicted_label)
    ground_truth = getattr(tree, ground_truth_label)

    # Get the depth of the predicted terminal label
    terminal_depth = node.depth

    # Get the confidence of the terminal label
    confidence = node.posteriors[sq]

    # Get the depth of the lowest common subsumer of the terminal prediction and ground truth
    lowest_common_subsumer_depth = lowest_common_subsumer(node, ground_truth).depth

    # Loop until we get a confident label
    while confidence < confidence_threshold:
        # Get the parent node
        node = node.parent

        # Get all terminal nodes under this ancestor
        terminals = node.leaves

        # Get the sum of all terminal softmax values
        s = 0
        for terminal in terminals:
            s += terminal.softmax
        
        # Quantize the softmax values into a histogram bin
        sq = quantize_softmax(s, nbins=nbins)

        # Get the confidence for this node
        confidence = node.posteriors[sq]

    # Get the final prediction node, index, and label
    final_prediction_index = node.index
    final_prediction_label = node.label

    # Determine if the final prediction is an ancestor of the ground truth
    if node in ground_truth.ancestors:
        is_subsumer = True
    else:
        is_subsumer = False

    if node in ground_truth.path:
        is_path = True
    else:
        is_path = False

    final_depth = node.depth 

    return (ground_truth_index, ground_truth_label, 
            predicted_index, predicted_label, 
            final_prediction_index, final_prediction_label, 
            node, is_subsumer, is_path,
            terminal_depth, final_depth, lowest_common_subsumer_depth,
            confidence)

def quantize_softmax(s, nbins=10):
    """
    Quantizes a softmax score (i.e., finds a respective histogram bin)
    """
    return min(math.floor(s * nbins), nbins - 1)