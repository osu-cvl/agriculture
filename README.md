## Confidence-Driven Hierarchical Classification of Cultivated Plant Stresses

This repo contains a condensed version of the code used in the paper [Confidence-Driven Hierarchical Classification of Cultivated Plant Stresses](https://openaccess.thecvf.com/content/WACV2021/papers/Frank_Confidence-Driven_Hierarchical_Classification_of_Cultivated_Plant_Stresses_WACV_2021_paper.pdf) by [Logan Frank](https://loganfrank.github.io/), Christopher Wiegman, [Dr. Jim Davis](http://web.cse.ohio-state.edu/~davis.1719/), and Dr. Scott Shearer.

The contents in this repo are organized as follows:
* [base_classifiers](base_classifiers/) contains all code pertaining to the base classifiers used in this paper, with network architectures defined in the [architectures subdirectory](base_classifiers/architectures/).
* [davis](davis/) contains all code used to reimplement the paper [Hierarchical Semantic Labeling with Adaptive Confidence](http://web.cse.ohio-state.edu/~davis.1719/Publications/isvc19a.pdf).
* [hedging](hedging/) contains all code used to reimplement the paper [Hedging Your Bets: Optimizing Accuracy-Specificity Trade-offs in Large Scale Visual Recognition](http://vision.stanford.edu/documents/DengKrauseBergFei-Fei_CVPR2012.pdf)
* [trees](trees/) contains example hierarchy txt files
* [utils](utils/) contains some utility functions and classes

### Code

To use our code begin with installing the necessary requirements, assuming you already have some environment with Python 3.7 and pip: ```pip install -r requirements.txt```

You may adapt and use the code provided for training your own base classifier (run everything from the top-level of this repo):

```
python base_classifiers/train.py \
    --image_dir <path-to-your-dataset-according-to-pytorch-image-folder-class> \
    --network_dir <wherever-you-want-to-save-your-network-related-data> \
    --results_dir <wherever-you-want-to-save-your-results-from-training> \
    --dataset <name-of-your-dataset> \
    --name <whatever-you-want-to-name-your-experiment> \
    --network <whichever-network-you-want-to-use> \
    --batch_size <some-int> \
    --learning_rate <some-float> \
    --num_epochs <some-int> \
    --balance_dataset <whether-or-not-you-want-to-balance-the-training-dataset>
```

Currently, our code only supports the ResNet-18 and small CNN networks used in our paper. More can easily be added. If you choose to balance the training dataset (using instance replication), it is strongly recommended you choose a batch size that is a multiple of your number of training classes so there is the same number of examples from each class in every batch.

Once a base classifier has been trained, the approach we applied to our datasets can be ran with:

```
python davis/pipeline.py \
    --image_dir <path-to-your-dataset-according-to-pytorch-image-folder-class> \
    --network_path <path-to-your-base-classifier-pth-file> \
    --tree_path <path-to-your-hierarchy-txt-file> \
    --results_dir <wherever-you-want-to-save-your-results-from-training> \
    --dataset <name-of-your-dataset> \
    --name <whatever-you-want-to-name-your-experiment> \
    --network <whichever-network-you-want-to-use-as-your-base-classifier> \
    --nbins <number-of-desired-histogram-bins-for-calibration> \
    --priors <what-type-of-priors> \
    --confidence <some-float-for-confidence-threshold>
```

By default, we run with nbins == 10, priors == 'equal' ('data' or 'manual' can also be used), and confidence == 0.5 (confidence should be [0, 1]).
