## Deep learning and array processing libraries
import numpy as np 
import torch
import torch.nn.functional as F 

## Inner-Project Imports
from davis_utils.estimation_utils import *
from davis_utils.tree import load_tree_from_file
from davis_utils.tree import update_softmax
from davis_utils.tree import update_priors


def estimation(tree_path, network, compute_device, train_dataloader, val_dataloader, nbins, priors):
    """
    Performs the estimation phase of "Hierarchical Semantic Labeling with Adaptive Confidence"
    """

    # Get the classes in the dataset
    classes = train_dataloader.dataset.classes 

    # Load the tree from a txt file
    tree = load_tree_from_file(tree_path, nbins=nbins)

    # Compute the priors (using training dataset if necessary)
    if priors == 'equal':
        positive_priors, negative_priors = get_equal_priors(train_dataloader)
    elif priors == 'data':
        positive_priors, negative_priors = get_priors(train_dataloader)
    elif priors == 'manual':
        positive_priors, negative_priors = get_handtuned_priors(train_dataloader)
    
    # Update the priors in the tree
    update_priors(tree.unknown, positive_priors, negative_priors)

    # Get the set of all nodes in the tree
    all_nodes = set(tree.__dict__.keys())

    for index, sample in enumerate(val_dataloader):
        # Load in batch image data
        instance_data = sample[0]
        instance_data.requires_grad = False

        # Load in batch label data
        label_data = sample[1].item()

        # Send image and label data to device
        instance_data = instance_data.to(compute_device)

        # Forward pass and get the output predictions
        logits = network(instance_data).squeeze()

        # Get the softmax vector
        softmaxes = F.softmax(logits, dim=0).detach().cpu().numpy()

        # Update the softmaxes on the tree
        update_softmax(tree.unknown, softmaxes)

        # Get the softmax value for the flat prediction
        s = softmaxes[label_data] 

        # Quantize the neural network softmax value into a histogram bin
        sq = quantize_softmax(s, nbins=nbins)

        # Get label of neural network flat prediction
        label_data_string = classes[label_data]

        # Get the TreeNode of that label
        label_node = getattr(tree, label_data_string)

        # Update the positive histogram of the guess node
        label_node.positive_hist[sq] += 1

        positive_nodes = set()
        positive_nodes.add(label_node.label)

        for ancestor in label_node.ancestors:
            # Add ancestor to the positive set
            positive_nodes.add(ancestor.label)

            # Get all terminal nodes under this ancestor
            terminals = ancestor.leaves

            # Get the sum of all terminal softmax values
            s = 0
            for terminal in terminals:
                s += terminal.softmax
            
            # Quantize the softmax values into a histogram bin
            sq = quantize_softmax(s, nbins=nbins)

            # Increment the positive histogram
            ancestor.positive_hist[sq] += 1

        for d in (all_nodes - positive_nodes):
            # Get the negative node
            node = getattr(tree, d)

            # Get all terminal nodes under negative node d
            terminals = node.leaves

            # Get the sum of all terminal softmax values
            s = 0
            for terminal in terminals:
                s += terminal.softmax
            
            # Quantize the softmax values into a histogram bin
            sq = quantize_softmax(s, nbins=nbins)

            # Increment the negative histogram bin
            node.negative_hist[sq] += 1
    
    for node_label in all_nodes:
        # Get a node in the tree
        node = getattr(tree, node_label)

        # L1 normalize the positive and negative histograms into likelihoods
        node.positive_hist = np.nan_to_num((node.positive_hist / sum(node.positive_hist)).astype(np.float32))
        node.negative_hist = np.nan_to_num((node.negative_hist / sum(node.negative_hist)).astype(np.float32))

    # Compute the posteriors P(l | sq) for all q using Bayes with likelihoods and priors
    compute_posteriors(tree, all_nodes)

    return tree