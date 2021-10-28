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

def compute_posteriors(tree, labels):
    """
    Compute posterior probabilities using Bayes' Rule for every node in a tree
    """
    for label in labels:
        node = getattr(tree, label)
        node.posteriors = np.nan_to_num((node.positive_hist * node.positive_prior) / ((node.positive_hist * node.positive_prior) + (node.negative_hist * node.negative_prior)))

def quantize_softmax(s, nbins=10):
    """
    Quantizes a softmax score (i.e., finds a respective histogram bin)
    """
    return min(math.floor(s * nbins), nbins - 1)

def get_priors(train_dataloader):
    """
    Calculates the priors based off the training dataset
    """
    instances = np.array(train_dataloader.dataset.targets)
    class_indexes, counts = np.unique(instances, return_counts=True)

    total_num_instances = len(instances)
    positive_priors = counts / total_num_instances
    negative_priors = 1 - positive_priors

    return positive_priors, negative_priors

def get_equal_priors(train_dataloader):
    """
    Calculate equal priors based off the number of classes
    """
    num_classes = len(train_dataloader.dataset.classes)
    individual_prior = 1 / num_classes
    positive_priors = np.repeat(individual_prior, num_classes)
    negative_priors = 1 - positive_priors

    return positive_priors, negative_priors

def get_handtuned_priors(train_dataloader):
    """
    Get priors manually set by a human
    """
    classes = train_dataloader.dataset.classes

    positive_priors = np.zeros(len(classes))
    negative_priors = np.zeros(len(classes))
    for index, value in enumerate(classes):
        positive_priors[index] = float(input(f'Enter a prior for {value}: '))
        negative_priors[index] = 1 - positive_priors[index]

    if sum(positive_priors) > 1:
        raise Exception("Sum of priors exceeds 1")

    return positive_priors, negative_priors
