import math
import numpy as np

def balance(dataset):

    instances = np.array(dataset.samples)
    file_names = instances[:, 0]
    instance_labels = instances[:, 1].astype(np.int16)

    classes, class_counts = np.unique(dataset.targets, return_counts=True)
    num_classes = len(classes)
    largest_num_samples = max(class_counts)

    instances = np.empty((num_classes, largest_num_samples), dtype=object)
    labels = np.empty((num_classes, largest_num_samples), dtype=np.int16)

    for class_index in classes:
        indexes = np.argwhere(instance_labels == class_index).flatten()
        class_files = file_names[indexes]
        class_labels = instance_labels[indexes]

        class_num_samples = len(class_files)

        shuffle = np.arange(class_num_samples)
        np.random.shuffle(shuffle)

        class_files = class_files[shuffle]
        class_labels = class_labels[shuffle]

        multiplier = math.ceil(largest_num_samples / class_num_samples)

        class_files = np.tile(class_files, multiplier)
        class_labels = np.tile(class_labels, multiplier)

        class_files = class_files[:largest_num_samples]
        class_labels = class_labels[:largest_num_samples]

        instances[class_index, :] = class_files
        labels[class_index, :] = class_labels
    
    # Guarantee one instance of each class is in every batch
    instances = np.transpose(instances).flatten()
    labels = np.transpose(labels).flatten()

    instance_to_label = np.vstack((instances, labels)).T
    instance_to_label = list(map(tuple, instance_to_label))

    dataset.imgs = instance_to_label
    dataset.samples = instance_to_label
    dataset.targets = labels

    return dataset