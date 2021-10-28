import os

## Deep learning and array processing libraries
import numpy as np 
np.set_printoptions(suppress=True)
import pandas as pd 
import matplotlib as mpl 
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import metrics

## Inner-Project Imports
from davis_utils.inference_utils import process_instance
from utils.metrics import *

def inference(confidence_threshold, experiment, tree, network, compute_device, dataloader, nbins, results_directory, save_results=True):
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
    is_paths = np.full(len(dataloader.dataset), False, dtype=bool)
    is_subsumers = np.full(len(dataloader.dataset), False, dtype=bool)
    terminal_depths = np.zeros(len(dataloader.dataset), dtype=int)
    final_depths = np.zeros(len(dataloader.dataset), dtype=int)
    lcs_depths = np.zeros(len(dataloader.dataset), dtype=int)
    posteriors = np.zeros(len(dataloader.dataset))

    for index, sample in enumerate(dataloader):
        # Process a sample through the hierarchy
        sample_information = process_instance(confidence_threshold, tree, network, compute_device, sample, classes, nbins)

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

    # Save the results if desired
    if save_results:
        with open(os.path.abspath(f'{results_directory}{experiment}_results_test_conf{confidence_threshold}.txt'), 'w') as f:
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

        # Create a confusion matrix using all classes and the base classifier
        confusion_matrix = metrics.confusion_matrix(gt_indexes, predicted_indexes)
        confusion_matrix = (confusion_matrix.T / np.sum(confusion_matrix, axis=1)).T
        cm_df = pd.DataFrame(confusion_matrix, index=dataloader.dataset.classes, columns=dataloader.dataset.classes)
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        sn.heatmap(cm_df, cmap='YlGnBu', cbar=False, annot=True, ax=ax)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        ax.set_title('Test Set Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        fig.savefig(os.path.abspath(f'{results_directory}{experiment}_confusion_matrix_test_base.png'))

        # Create a binary confusion matrix using healthy and "Stressed" and the base classifier
        binary_confusion_matrix = metrics.confusion_matrix(ground_truth_healthy, predicted_healthy)
        binary_confusion_matrix = (binary_confusion_matrix.T / np.sum(binary_confusion_matrix, axis=1)).T
        cm_df_bin = pd.DataFrame(binary_confusion_matrix, index=[['stressed', 'healthy']], columns=['stressed', 'healthy'])
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        sn.heatmap(cm_df_bin, cmap='YlGnBu', cbar=False, annot=True, ax=ax)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        ax.set_title('Test Set Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        fig.savefig(os.path.abspath(f'{results_directory}{experiment}_confusion_matrix_test_binary.png'))

        with open(os.path.abspath(f'{results_directory}{experiment}_results_test_conf{confidence_threshold}.txt'), 'a') as f:
            f.write(f'Normal Confusion Matrix: \n{np.round(confusion_matrix, decimals=4)} \n')
            f.write('---\n')
            f.write(f'Binary Confusion Matrix: \n{np.round(binary_confusion_matrix, decimals=4)} \n')