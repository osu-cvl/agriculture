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

# Other imports
import sklearn.metrics as metrics

## Inter-project imports
from architectures import small_cnn, resnet18
from utils import parameters
from utils.transforms import *
from utils import balance_dataset
from utils.validate import validation
from utils.recorder import TrainingRecorder

def arguments():
    parser = argparse.ArgumentParser(description='Training baseline classifier')
    parser.add_argument('--image_dir', '-I', 
                        default='', type=str, 
                        metavar='ID', help='location of image data')
    parser.add_argument('--network_dir', '-N', 
                        default='', type=str, 
                        metavar='ND', help='location to save network data')
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
    parser.add_argument('--learning_rate', '-l', 
                        default=0.001, type=float, 
                        metavar='LR', help='learning rate')
    parser.add_argument('--weight_decay', '-w', 
                        default=0.0001, type=float, 
                        metavar='WD', help='weight decay')
    parser.add_argument('--momentum', '-m', 
                        default=0.9, type=float, 
                        metavar='M', help='momentum')
    parser.add_argument('--num_epochs', '-e', 
                        default=30, type=int, 
                        metavar='NE', help='number of epochs to train for')
    parser.add_argument('--balance_dataset',
                        default='True', type=str,
                        metavar='BAL', choices=['True', 'False'],
                        help='do we balance the training dataset')
    parser.add_argument('--continue_training', '--cont', 
                        default='False', type=str, 
                        metavar='CONT', choices=['True', 'False'], 
                        help='are we continuing training')
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
    assert args['network_dir'] != '', f'Must provide a path to a network directory for {args["dataset"]}'
    assert args['results_dir'] != '', f'Must provide a path to a results directory for {args["dataset"]}'

    # Determine if we are continuing training from a certain epoch
    args['continue_training'] = (args['continue_training'] == 'True')

    # Create the experiment name
    args['name'] = f'{args["dataset"]}_train' if args['name'] == '' else args['name']   

    # Ensure we don't accidentally overwrite anything by checking how many previous experiments share the same name
    if not args['continue_training']:
        directories = [name for name in os.listdir(os.path.abspath(args['network_dir'])) if os.path.isdir(f'{args["network_dir"]}{name}') and args['name'] in name]
        num = len(directories)
        args['name'] = f'{args["name"]}_{num}'

    # Determine if we will be balancing the training dataset using instance replication
    args['balance_dataset'] = (args['balance_dataset'] == 'True')

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

    # Set up a parameters object for saving hyperparameters, etc.
    parameters = parameters.Parameters(args['name'], **args)
    if args['continue_training']:
        with open(os.path.abspath(f'{args["network_dir"]}{args["name"]}_parameters.pkl'), 'rb') as f:
            parameters = pickle.load(f)

    # Create the data transforms for each respective set
    if args['dataset'] == 'tomato':
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
                                        RotationTransform(angles=[0, 90, 180, 270]), GammaJitter(low=0.9, high=1.1), 
                                        BrightnessJitter(low=0.9, high=1.1), transforms.ToTensor(), 
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    elif args['dataset'] == 'corn' or args['dataset'] == 'soybean':
        train_transform = transforms.Compose([AdaptiveCenterCrop(), Resize(size=345), 
                                        transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), 
                                        RotationTransform(angles=[0, 90, 180, 270]), GammaJitter(low=0.9, high=1.1), 
                                        BrightnessJitter(low=0.9, high=1.1), transforms.ToTensor(), 
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])                         
        test_transform = transforms.Compose([AdaptiveCenterCrop(), Resize(size=345), 
                                        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    else:
        raise Exception('Dataset not yet supported')

    # Create the data sets
    train_dataset = datasets.ImageFolder(root=f'{args["image_dir"]}train/', transform=train_transform)
    val_dataset = datasets.ImageFolder(root=f'{args["image_dir"]}val/', transform=test_transform)

    # Optionally balance the training dataset through instance replication
    if args['balance_dataset']:
        train_dataset = balance_dataset.balance(train_dataset)

    # Create the DataLoaders from the data sets
    train_dataloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=(not args['balance_dataset']))
    val_dataloader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

    # Create the network, (potentially) load network state dictionary, and send the network to the compute device
    num_classes = len(train_dataset.classes)
    if args['network'] == 'small_cnn':
        network = small_cnn.SmallCNN(num_classes)
    elif args['network'] == 'resnet18':
        network = resnet18.ResNet18(num_classes)

    if args['continue_training']:
        # Load some weights from a previous training session
        network.load_state_dict(torch.load(os.path.abspath(f'{args["network_dir"]}{args["name"]}/{args["name"]}.pth'), map_location='cpu'))
    else:
        # Make the directory if it doesn't exist
        if not os.path.exists(os.path.abspath(f'{args["network_dir"]}{args["name"]}')): 
            os.makedirs(os.path.abspath(f'{args["network_dir"]}{args["name"]}'))

        # Save the initial weights
        torch.save(network.state_dict(), os.path.abspath(f'{args["network_dir"]}{args["name"]}/{args["name"]}_initial_weights.pth'))
    
    # Ensure all parameters allow for gradient descent
    for parameter in network.parameters():
        parameter.requires_grad = True

    # Send to device (preferably GPU)
    network = network.to(args['device'])

    # Create the optimizer and (potentially) load the optimizer state dictionary
    optimizer = optim.SGD(network.parameters(), lr=args['learning_rate'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    if args['continue_training']:
        optimizer.load_state_dict(torch.load(os.path.abspath(f'{args["network_dir"]}{args["name"]}_optimizer.pth')))

    # Create a step learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Create classification loss function
    classification_loss_func = nn.CrossEntropyLoss()

    # Create a recorder
    recorder = TrainingRecorder(args['results_dir'], args['network_dir'], parameters, new=(parameters.epoch == 0))

    # Begin training the neural network
    for epoch in range(parameters.epoch, parameters.num_epochs):
        true_classes = np.empty(0)
        predicted_classes = np.empty(0)

        running_loss = 0.0
        network.train()
        for batch_num, batch_sample in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Load in batch image data
            image_data = batch_sample[0]
            image_data.requires_grad = False

            # Load in batch label data
            label_data = batch_sample[1]
            label_data.requires_grad = False

            # If we balanced the dataset, then we need to randomize the batch
            if args['balance_dataset']:
                shuffle = torch.randperm(label_data.shape[0])
                label_data = label_data[shuffle]
                image_data = image_data[shuffle]

            # Send image and label data to device
            image_data = image_data.to(args['device'])
            label_data = label_data.to(args['device'])

            # Forward pass and get the output predictions
            predictions = network(image_data)

            # Pass through loss function and perform back propagation
            loss = classification_loss_func(predictions, label_data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            print(f'Experiment: {parameters.experiment} -- Epoch: {epoch} -- Batch: {batch_num} -- Loss: {loss.item()}')

            # Record the actual and predicted labels for the instance
            true_classes = np.concatenate((true_classes, label_data.detach().cpu().numpy()))
            _, predictions = torch.max(predictions, 1)
            predicted_classes = np.concatenate((predicted_classes, predictions.detach().cpu().numpy()))

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()

        # Get the training accuracy
        loss = running_loss / (batch_num + 1)
        accuracy = metrics.accuracy_score(true_classes, predicted_classes)

        # Check the validation error after each training epoch
        print(f'Evaluating validation set (epoch {epoch}):')
        val_loss, val_accuracy = validation(network=network, dataloader=val_dataloader, compute_device=args['device'], 
                                                        experiment=parameters.experiment, results_directory=args['results_dir'], 
                                                        classification_loss_func=classification_loss_func)

        recorder.record(epoch, loss, val_loss, accuracy, val_accuracy)
        recorder.update(epoch, val_loss, network.state_dict(), optimizer.state_dict())