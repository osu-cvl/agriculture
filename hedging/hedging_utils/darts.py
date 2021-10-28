import numpy as np
from hedging_utils.tree import update_posteriors
from hedging_utils.tree import update_inner_posteriors
from hedging_utils.tree import info_gain

def compute_posteriors(logits, tree, terminals, calibrated_models):
    posteriors = np.zeros(logits.shape, dtype=float)

    # Look through all terminal classes
    for terminal in terminals:
        # Get the calibrated probabilities for a certain class
        probability = calibrated_models[terminal].predict_proba(logits[:, terminal].reshape((-1, 1)))
        
        # Get the probability for True
        probability = probability[:, 1]

        # Add to posteriors matrix
        posteriors[:, terminal] = probability

    # Normalize each instance
    posteriors = posteriors / np.sum(posteriors, axis=1).reshape((-1, 1))

    return posteriors

def process_instance(nodes, index, lamb):
    weighted_posterior = 0

    # Loop through all nodes and pick the one with the best weighted posterior
    for node in nodes:
        node_weighted_posterior = (node.info_gain + lamb) * node.posteriors[index]
        if node_weighted_posterior > weighted_posterior:
            weighted_posterior = node_weighted_posterior
            best_node = node

    return best_node
    

def predict_hedging(tree, terminals, logits, calibrated_models, lamb, posteriors=None):
    num_instances = len(logits)
    predictions = np.full(num_instances, '', dtype=object)
    confidences = np.zeros(num_instances)

    if posteriors is None:
        # compute the posteriors if not already provided
        posteriors = compute_posteriors(logits, tree, terminals, calibrated_models)

    # Get a set of all nodes in the tree
    nodes = list(tree.__dict__.values())

    for index in range(num_instances):
        node = process_instance(nodes, index, lamb)
        predictions[index] = node.label
        confidences[index] = node.posteriors[index]

    return predictions.astype('U'), confidences

def compute_accuracy(tree, ground_truth, predictions):
    """
    Computes hierarchical accuracy
    """
    num_correct = 0
    num_instances = len(ground_truth)
    for index in range(num_instances):
        gt_node = getattr(tree, ground_truth[index])
        pred_node = getattr(tree, predictions[index])

        if gt_node in pred_node.descendants or gt_node == pred_node:
            num_correct += 1

    return num_correct / num_instances

def check_lambda(tree, terminals, logits, ground_truth, posteriors, calibrated_models, confidence_threshold,
                    confidence_tolerance, lambda_upper_bound, lambda_lower_bound, current_lambda):
    """
    Checks if the lambda is sufficient
    """
    predictions, _ = predict_hedging(tree, terminals, logits, calibrated_models, current_lambda, posteriors)
    classifier_accuracy = compute_accuracy(tree, ground_truth, predictions)
    if abs(classifier_accuracy - confidence_threshold) <= confidence_tolerance:
        return 0, classifier_accuracy
    elif classifier_accuracy > confidence_threshold:
        return -1, classifier_accuracy
    else:
        return 1, classifier_accuracy

def binary_search(tree, terminals, logits, ground_truth, posteriors, calibrated_models, 
                    confidence_threshold, confidence_tolerance, maximum_iterations, lambda_upper_bound, lambda_lower_bound):
    """
    Performs a binary search to find the best lambda
    """
    lambda_lower_bound_history = [lambda_lower_bound]
    lambda_upper_bound_history = [lambda_upper_bound]
    current_lambda = (lambda_upper_bound + lambda_lower_bound) / 2
    lambda_history = [current_lambda]
    check, classifier_accuracy = check_lambda(tree, terminals, logits, ground_truth, posteriors, 
                                                calibrated_models, confidence_threshold, confidence_tolerance,
                                                lambda_upper_bound, lambda_lower_bound, current_lambda)
    classifier_accuracy_history = [classifier_accuracy]

    for _ in range(maximum_iterations):
        if check == 0:
            return np.array(lambda_lower_bound_history), np.array(lambda_upper_bound_history), np.array(lambda_history), np.array(classifier_accuracy_history)
        if check == -1:
            lambda_upper_bound = current_lambda
        else:
            lambda_lower_bound = current_lambda
        lambda_lower_bound_history.append(lambda_lower_bound)
        lambda_upper_bound_history.append(lambda_upper_bound)
        current_lambda = (lambda_upper_bound + lambda_lower_bound) / 2
        lambda_history.append(current_lambda)
        check, classifier_accuracy = check_lambda(tree, terminals, logits, ground_truth, posteriors, 
                                                calibrated_models, confidence_threshold, confidence_tolerance,
                                                lambda_upper_bound, lambda_lower_bound, current_lambda)
        classifier_accuracy_history.append(classifier_accuracy)
    return np.array(lambda_lower_bound_history), np.array(lambda_upper_bound_history), np.array(lambda_history), np.array(classifier_accuracy_history)


def darts(tree, terminals, logits, ground_truth, calibrated_models, max_reward, epsilon, confidence_tolerance):
    """
    Performs DARTS algorithm to find the best lambda
    """
    confidence_threshold = 1 - epsilon
    min_reward = 0

    # Compute posteriors
    posteriors = compute_posteriors(logits, tree, terminals, calibrated_models)

    # Update posteriors in the tree
    update_posteriors(tree.unknown, posteriors)
    update_inner_posteriors(tree)

    # Update info gain in the tree
    info_gain(tree.unknown, len(terminals))

    # Set lambda to 0
    lamb = 0

    # Get f0 predictions
    initial_predictions, _ = predict_hedging(tree, terminals, logits, calibrated_models, lamb, posteriors)

    # Compute f0 accuracy
    initial_accuracy = compute_accuracy(tree, ground_truth, initial_predictions)

    # Check if f0 is good enough
    if initial_accuracy >= confidence_threshold:
        print(f'lambda = {lamb} when confidence_threshold = {confidence_threshold}')
        return lamb
    
    # Determine upper bound of lambda
    lambda_upper_bound = ((max_reward * confidence_threshold) - min_reward) / epsilon
    lambda_lower_bound = 0

    # Define the maximum number of iterations
    maximum_iterations = 100

    # Do a binary search to find optimal lambda
    binary_search_results = binary_search(tree, terminals, logits, ground_truth, posteriors, calibrated_models, 
                    confidence_threshold, confidence_tolerance, maximum_iterations, lambda_upper_bound, lambda_lower_bound)
    lambda_lower_bound_history, lambda_upper_bound_history, lambda_history, classifier_accuracy_history = binary_search_results

    return lambda_history[-1]