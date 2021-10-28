## Basic Python libraries
import sys
import os
import math
sys.path.append(os.getcwd() + '/')

## Deep learning and array processing libraries
import numpy as np 
np.set_printoptions(suppress=True)
import pandas as pd 
import matplotlib as mpl 
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics

## Inner-Project Imports
from utils.metrics import *
from hedging_utils.darts import compute_posteriors
from hedging_utils.darts import compute_accuracy
from hedging_utils.tree import update_posteriors
from hedging_utils.tree import update_inner_posteriors
from hedging_utils.tree import lowest_common_subsumer

def process_instance(tree, terminals, network, calibrated_models, compute_device, sample, classes, lamb):
    # Load in batch image data
    image_data = sample[0]
    image_data.requires_grad = False
    image_data = image_data.to(compute_device)

    # Load in batch label data
    ground_truth_index = sample[1].item()

    # Forward pass and get the output predictions
    logits = network(image_data).squeeze()
    logits = logits.reshape((1, -1))

    # Get the softmax vector
    posteriors = compute_posteriors(logits.detach().cpu().numpy(), tree, terminals, calibrated_models)

    # Get the classifiers predictions
    predicted_index = np.argmax(posteriors)

    # Update the softmaxes on the tree
    update_posteriors(tree.unknown, posteriors)
    update_inner_posteriors(tree)

    # Get label of neural network flat prediction
    predicted_label = classes[predicted_index]
    ground_truth_label = classes[ground_truth_index]

    # Get the TreeNode of that label
    node = getattr(tree, predicted_label)
    predicted = getattr(tree, predicted_label)
    ground_truth = getattr(tree, ground_truth_label)

    # Get the depth of the predicted terminal label
    terminal_depth = node.depth

    # Get the depth of the lowest common subsumer of the terminal prediction and ground truth
    lowest_common_subsumer_depth = lowest_common_subsumer(node, ground_truth).depth

    # Get a set of all nodes in the tree
    nodes = list(tree.__dict__.values())

    # What is the best weighted posterior?
    weighted_posterior = 0

    # Loop through all nodes and pick the one with the best weighted posterior
    for potential_node in nodes:
        node_weighted_posterior = (potential_node.info_gain + lamb) * potential_node.posteriors[0]
        if node_weighted_posterior > weighted_posterior:
            weighted_posterior = node_weighted_posterior
            node = potential_node

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
            node.posteriors[0])

def inference(experiment, tree, terminals, network, calibrated_models, compute_device, dataloader, lamb, confidence_threshold, results_directory, save_results=True, return_results=False):
    classes = dataloader.dataset.classes

    # Create a lot of empty arrays for holding results and other information for computing scores
    gt_labels = np.full(len(dataloader.dataset), '', dtype=object)
    gt_indexes = np.zeros(len(dataloader.dataset))
    predicted_labels = np.full(len(dataloader.dataset), '', dtype=object)
    predicted_indexes = np.zeros(len(dataloader.dataset))
    final_labels = np.full(len(dataloader.dataset), '', dtype=object)
    final_indexes = np.zeros(len(dataloader.dataset))
    is_terminals = np.full(len(dataloader.dataset), False, dtype=bool)
    num_terminals = np.zeros(len(dataloader.dataset))
    is_roots = np.full(len(dataloader.dataset), False, dtype=bool)
    confidences = np.zeros(len(dataloader.dataset))
    is_paths = np.full(len(dataloader.dataset), False, dtype=bool)
    is_subsumers = np.full(len(dataloader.dataset), False, dtype=bool)
    terminal_depths = np.zeros(len(dataloader.dataset), dtype=int)
    final_depths = np.zeros(len(dataloader.dataset), dtype=int)
    lcs_depths = np.zeros(len(dataloader.dataset), dtype=int)
    posteriors = np.zeros(len(dataloader.dataset))

    for index, sample in enumerate(dataloader):
        # Process a sample through the hierarchy
        sample_information = process_instance(tree, terminals, network, calibrated_models, compute_device, sample, classes, lamb)

        # Parse out the sample information necessary for hierarchy metrics
        gt_indexes[index] = sample_information[0]
        gt_labels[index] = sample_information[1]
        predicted_indexes[index] = sample_information[2]
        predicted_labels[index] = sample_information[3]
        final_indexes[index] = sample_information[4]
        final_labels[index] = sample_information[5]
        is_terminals[index] = sample_information[6].is_leaf
        num_terminals[index] = len(sample_information[6].leaves)
        is_roots[index] = sample_information[6].is_root
        confidences[index] = sample_information[6].posteriors
        is_subsumers[index] = sample_information[7]
        is_paths[index] = sample_information[8]
        terminal_depths[index] = sample_information[9]
        final_depths[index] = sample_information[10]
        lcs_depths[index] = sample_information[11]
        posteriors[index] = sample_information[12]

    # Convert object arrays to string arrays
    gt_labels = gt_labels.astype('U') 
    predicted_labels = predicted_labels.astype('U')
    final_labels = final_labels.astype('U')

    # Determine binary classification results
    predicted_healthy = predicted_labels == 'healthy'
    ground_truth_healthy = gt_labels == 'healthy'
    stressed = np.logical_not(predicted_healthy) & np.logical_not(ground_truth_healthy)

    # Determine set of initially correct and incorrect predictions
    sc = predicted_indexes == gt_indexes
    sic = predicted_indexes != gt_indexes

    # Determine post-inference correct results
    correct = sc | is_subsumers

    # Set of initially correct predictions that were softened
    s_soft = ((final_indexes != predicted_indexes) & (sc))

    # Set of initially incorrect predictions that were reformed
    s_ref = ((final_indexes != predicted_indexes) & (sic))

    # Calculate all the different hierarchical metric scores
    c_p = c_persist(sc, is_terminals)
    w_c_p = weighted_c_persist(gt_indexes, sc, is_terminals)
    c_w = c_withdrawn(sc, is_roots)
    w_c_w = weighted_c_withdrawn(gt_indexes, sc, is_roots)
    c_s = c_soften(c_p, c_w)
    w_c_s = weighted_c_soften(w_c_p, w_c_w)
    c_c = c_corrupt(sc, is_paths)
    w_c_c = weighted_c_corrupt(gt_indexes, sc, is_paths)
    ic_rem = ic_remain(sic, is_subsumers, is_roots)
    w_ic_rem = weighted_ic_remain(gt_indexes, sic, is_subsumers, is_roots)
    ic_w = ic_withdrawn(sic, is_roots)
    w_ic_w = weighted_ic_withdrawn(gt_indexes, sic, is_roots)
    ic_ref = ic_reform(ic_rem, ic_w) 
    w_ic_ref = weighted_ic_reform(w_ic_rem, w_ic_w) 
    ig = info_gain(sc, len(dataloader.dataset.classes), num_terminals)
    w_ig = weighted_info_gain(gt_indexes, sc, len(dataloader.dataset.classes), num_terminals)
    p = np.mean(posteriors) 
    w_p = weighted_posteriors(gt_indexes, posteriors) 
    w_v = weighted_valid(gt_indexes, is_roots)
    w_s_s = weighted_specified_stress(gt_indexes, stressed, final_labels)

    print(f'{"C-Persist:" : <20} {c_p : 0.4f}')
    print(f'{"C-Withdrawn:" : <20} {c_w : 0.4f}')
    print(f'{"C-Soften:" : <20} {c_s : 0.4f}')
    print(f'{"C-Corrupt:" : <20} {c_c : 0.4f}')
    print(f'{"IC-Remain:" : <20} {ic_rem : 0.4f}')
    print(f'{"IC-Withdrawn:" : <20} {ic_w : 0.4f}')
    print(f'{"IC-Reform:" : <20} {ic_ref : 0.4f}')
    print(f'{"Info-Gain:" : <20} {ig : 0.4f}')
    print(f'{"Posteriors:" : <20} {p : 0.4f}')
    print('---')
    print(f'{"Weighted C-Persist:" : <20} {w_c_p : 0.4f}')
    print(f'{"Weighted C-Withdrawn:" : <20} {w_c_w : 0.4f}')
    print(f'{"Weighted C-Soften:" : <20} {w_c_s : 0.4f}')
    print(f'{"Weighted C-Corrupt:" : <20} {w_c_c : 0.4f}')
    print(f'{"Weighted IC-Remain:" : <20} {w_ic_rem : 0.4f}')
    print(f'{"Weighted IC-Withdrawn:" : <20} {w_ic_w : 0.4f}')
    print(f'{"Weighted IC-Reform:" : <20} {w_ic_ref : 0.4f}')
    print(f'{"Weighted Info-Gain:" : <20} {w_ig : 0.4f}')
    print(f'{"Weighted Posteriors:" : <20} {w_p : 0.4f}')
    print('---')
    print(f'{"% Valid (-root):" : <20} {np.sum(np.logical_not(is_roots)) / len(is_roots) * 100 : 0.4f}')
    print(f'{"% Valid (-root) (wtd):" : <20} {w_v : 0.4f}')
    print(f'{"% Correct:": <20} {np.sum(correct) / len(correct) * 100 : 0.4f}')
    print(f'{"% Correct (wtd)" : <20} {weighted_accuracy(gt_indexes, correct) * 100 : 0.4f}')
    print(f'{"% Defined Stresses:" : <20} {np.sum(stressed & (final_labels != "stressed") & (final_labels != "unknown")) / np.sum(stressed) * 100 : 0.4f}')
    print(f'{"% Defined Stresses (wtd):" : <20} {w_s_s : 0.4f}')

    if save_results:
        # Save the general accuracy, F1 score, and Cohen's Kappa score
        with open(os.path.abspath(f'{results_directory}{experiment}_conf{confidence_threshold}_hedging.txt'), 'w') as f:
            f.write(f'{"C-Persist:" : <20} {c_p : 0.4f}\n')
            f.write(f'{"C-Withdrawn:" : <20} {c_w : 0.4f}\n')
            f.write(f'{"C-Soften:" : <20} {c_s : 0.4f}\n')
            f.write(f'{"C-Corrupt:" : <20} {c_c : 0.4f}\n')
            f.write(f'{"IC-Remain:" : <20} {ic_rem : 0.4f}\n')
            f.write(f'{"IC-Withdrawn:" : <20} {ic_w : 0.4f}\n')
            f.write(f'{"IC-Reform:" : <20} {ic_ref : 0.4f}\n')
            f.write(f'{"Info-Gain:" : <20} {ig : 0.4f}\n')
            f.write(f'{"Posteriors:" : <20} {p : 0.4f}\n')
            f.write('---\n')
            f.write(f'{"Weighted C-Persist:" : <20} {w_c_p : 0.4f}\n')
            f.write(f'{"Weighted C-Withdrawn:" : <20} {w_c_w : 0.4f}\n')
            f.write(f'{"Weighted C-Soften:" : <20} {w_c_s : 0.4f}\n')
            f.write(f'{"Weighted C-Corrupt:" : <20} {w_c_c : 0.4f}\n')
            f.write(f'{"Weighted IC-Remain:" : <20} {w_ic_rem : 0.4f}\n')
            f.write(f'{"Weighted IC-Withdrawn:" : <20} {w_ic_w : 0.4f}\n')
            f.write(f'{"Weighted IC-Reform:" : <20} {w_ic_ref : 0.4f}\n')
            f.write(f'{"Weighted Info-Gain:" : <20} {w_ig : 0.4f}\n')
            f.write(f'{"Weighted Posteriors:" : <20} {w_p : 0.4f}\n')
            f.write('---\n')
            f.write(f'{"% Valid (-root):" : <20} {np.sum(np.logical_not(is_roots)) / len(is_roots) * 100 : 0.4f}\n')
            f.write(f'{"% Valid (-root) (wtd):" : <20} {w_v : 0.4f} \n')
            f.write(f'{"% Correct:": <20} {np.sum(correct) / len(correct) * 100 : 0.4f}\n')
            f.write(f'{"% Correct (wtd)" : <20} {weighted_accuracy(gt_indexes, correct) * 100 : 0.4f}\n')
            f.write(f'{"% Defined Stresses:" : <20} {np.sum(stressed & (final_labels != "stressed") & (final_labels != "unknown")) / np.sum(stressed) * 100 : 0.4f}\n')
            f.write(f'{"% Defined Stresses (wtd):" : <20} {w_s_s : 0.4f} \n')
            f.write('---\n')
    if return_results:
        return gt_labels, gt_indexes, predicted_labels, predicted_indexes, final_labels, final_indexes, posteriors