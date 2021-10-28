## Basic Python libraries
import sys
import os
import argparse
sys.path.append(os.getcwd() + '/')

## Deep learning and array processing libraries
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms 
import torchvision.datasets as datasets

## Inter-project imports
from architectures import small_cnn, resnet18
from utils import parameters
from utils.transforms import *
from utils import balance_dataset
from utils.validate import validation
from utils.recorder import EvaluateRecorder

def arguments():
    parser = argparse.ArgumentParser(description='Training baseline classifier')
    parser.add_argument('--image_dir', '-I', 
                        default='', type=str, 
                        metavar='ID', help='location of image data')
    parser.add_argument('--network_path', '-N', 
                        default='', type=str, 
                        metavar='NET', help='path to network pth file')
    parser.add_argument('--results_dir', '-R', 
                        default='', type=str, 
                        metavar='RD', help='location to save results')
    parser.add_argument('--dataset', '--data', '-d', 
                        default='tomato', type=str, 
                        metavar='DATA', help='name of data set')
    parser.add_argument('--name', 
                        default='', type=str, 
                        metavar='NAME', help='name of experiment')
    parser.add_argument('--network', '-n', 
                        default='small_cnn', type=str, 
                        metavar='N', choices=['small_cnn', 'resnet18'], 
                        help='network architecture')
    parser.add_argument('--batch_size', '-b', 
                        default=20, type=int, 
                        metavar='BS', help='batch size')
    parser.add_argument('--device', '--dev', 
                        default='cuda:0', type=str, 
                        metavar='DEV', choices=['cpu', 'cuda:0'], 
                        help='device id (e.g. \'cpu\', \'cuda:0\'')
    parser.add_argument('--seed', 
                        default=None, type=str, 
                        metavar='S', help='set a seed for reproducability')
    args = vars(parser.parse_args())

    # Ensure necessary paths have been provided
    assert args['image_dir'] != '', f'Must provide a path to an image directory for {args["dataset"]}'
    assert os.path.exists(args['network_path']), f'Must provide a path to a network pth file for {args["dataset"]}'
    assert args['results_dir'] != '', f'Must provide a path to a results directory for {args["dataset"]}'

    # Create the experiment name
    args['name'] = f'{args["dataset"]}_test' if args['name'] == '' else args['name']   

    # Ensure we don't accidentally overwrite anything by checking how many previous experiments share the same name
    directories = [name for name in os.listdir(os.path.abspath(args['results_dir'])) if os.path.isdir(f'{args["results_dir"]}{name}') and args['name'] in name]
    num = len(directories)
    args['name'] = f'{args["name"]}_{num}'

    # Set the device
    if 'cuda' in args['device']:
        assert torch.cuda.is_available(), 'Device set to GPU but CUDA not available'
    args['device'] = torch.device(args['device'])

    # Set a seed if provided
    if args['seed'] is not None:
        args['seed'] = int(args['seed'])
        torch.manual_seed(args['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args['seed'])

    return args

if __name__ == '__main__':
    # Get arguments
    args = arguments()

    # Create the data transforms for each respective set
    if args['dataset'] == 'tomato':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    elif args['dataset'] == 'corn' or args['dataset'] == 'soybean':                       
        transform = transforms.Compose([AdaptiveCenterCrop(), Resize(size=345), 
                                        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        raise Exception('Dataset not yet supported')

    # Create the data sets
    test_dataset = datasets.ImageFolder(root=f'{args["image_dir"]}test/', transform=transform)
    val_dataset = datasets.ImageFolder(root=f'{args["image_dir"]}val/', transform=transform)

    # Create the DataLoaders from the data sets
    test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    # Create the network, (potentially) load network state dictionary, and send the network to the compute device
    num_classes = len(val_dataset.classes)
    if args['network'] == 'small_cnn':
        network = small_cnn.SmallCNN(num_classes)
    elif args['network'] == 'resnet18':
        network = resnet18.ResNet18(num_classes)
    network.load_state_dict(torch.load(args['network_path'], map_location='cpu'))
    network.eval()

    # Send to device (preferably GPU)
    network = network.to(args['device'])

    # Get the batch size
    batch_size = args['batch_size']

    ### VALIDATION
    # Instantiate two arrays to keep track of the ground truth label and network prediction for each instance
    num_instances = len(val_dataset)
    true_classes = np.zeros(num_instances)
    predicted_classes = np.zeros(num_instances)

    with torch.no_grad():
        # Begin evaluating the neural network
        for batch_num, batch_sample in enumerate(val_dataloader):
            # Load in batch image data
            image_data = batch_sample[0]
            image_data.requires_grad = False

            # Load in batch label data
            label_data = batch_sample[1]
            label_data.requires_grad = False

            # Send image and label data to device
            image_data = image_data.to(args['device'])
            label_data = label_data.to(args['device'])

            # Forward pass and get the output predictions
            predictions = network(image_data)

            # Get the flat predictions
            _, predictions = torch.max(predictions, 1)

            # Record the actual and predicted labels for the instance
            true_classes[ batch_num * batch_size : min( (batch_num + 1) * batch_size, num_instances) ] = label_data.detach().cpu().numpy()
            predicted_classes[ batch_num * batch_size : min( (batch_num + 1) * batch_size, num_instances) ] = predictions.detach().cpu().numpy() 

    # Record results
    recorder = EvaluateRecorder(args['results_dir'], args['name'], 'val')
    accuracy = recorder.record(true_classes, predicted_classes, val_dataset.classes, save=True)

    ### TEST
    # Instantiate two arrays to keep track of the ground truth label and network prediction for each instance
    num_instances = len(test_dataset)
    true_classes = np.zeros(num_instances)
    predicted_classes = np.zeros(num_instances)

    with torch.no_grad():
        # Begin evaluating the neural network
        for batch_num, batch_sample in enumerate(test_dataloader):
            # Load in batch image data
            image_data = batch_sample[0]
            image_data.requires_grad = False

            # Load in batch label data
            label_data = batch_sample[1]
            label_data.requires_grad = False

            # Send image and label data to device
            image_data = image_data.to(args['device'])
            label_data = label_data.to(args['device'])

            # Forward pass and get the output predictions
            predictions = network(image_data)

            # Get the flat prediction
            _, predictions = torch.max(predictions, 1)

            # Record the actual and predicted labels for the instance
            true_classes[ batch_num * batch_size : min( (batch_num + 1) * batch_size, num_instances) ] = label_data.detach().cpu().numpy()
            predicted_classes[ batch_num * batch_size : min( (batch_num + 1) * batch_size, num_instances) ] = predictions.detach().cpu().numpy() 

    # Record results
    recorder = EvaluateRecorder(args['results_dir'], args['name'], 'test')
    accuracy = recorder.record(true_classes, predicted_classes, test_dataset.classes, save=True)
