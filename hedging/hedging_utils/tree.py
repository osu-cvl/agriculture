## Deep learning and array processing libraries
import numpy as np 
import pandas as pd 

## Tree-Related Imports
from anytree import NodeMixin, RenderTree

class GeneralTree(object):
    def __init__(self):
        pass
    
    def __repr__(self):
        return str(RenderTree(self.unknown))

class TreeNode(NodeMixin):
    def __init__(self, label, index, softmax=0, nbins=0, parent=None):
        super(TreeNode, self).__init__()
        self.label = label
        self.index = index
        self.softmax = softmax
        self.posteriors = np.empty(0, dtype=float)
        self.parent = parent
        self.info_gain = 0

    def set_parent(self, parent):
        if isinstance(parent, TreeNode):
            self.parent = parent
        else:
            raise Exception('parent must be of type TreeNode')

    def __eq__(self, other):
        if isinstance(other, TreeNode):
            return self.label == other.label
        else:
            raise Exception('other must be of type TreeNode')

    def __repr__(self):
        return f'[TreeNode: {self.label} / {self.index}]'

def update_posteriors(node, posteriors):
    """
    Recursively updates the posteriors values at terminal nodes in a tree.

    Inputs:
    node :: TreeNode
        the current node being processed
    softmax_values :: the array of softmax values for an example
    """
    if node.index >= 0:
        node.posteriors = posteriors[:, node.index]
    else:
        for child in node.children:
            update_posteriors(child, posteriors)

def update_inner_posteriors(tree):
    unknown = tree.unknown

    # Compute posteriors for unknown (root) node
    posteriors = np.zeros(unknown.leaves[0].posteriors.shape)
    for leaf in unknown.leaves:
        posteriors += leaf.posteriors
    unknown.posteriors = posteriors

    # Get the remaining non-terminal nodes
    for node in unknown.descendants:
        if node.index == -1:
            node_leaves = node.leaves 
            posteriors = np.zeros(node_leaves[0].posteriors.shape)
            for leaf in node_leaves:
                posteriors += leaf.posteriors
            node.posteriors = posteriors

def info_gain(node, num_terminals):
    node.info_gain = (np.log2(num_terminals) - np.log2(len(node.leaves))) / np.log2(num_terminals)
    for child in node.children:
            info_gain(child, num_terminals)

def update_softmax(node, softmax_values):
    """
    Recursively updates the softmax values at terminal nodes in a tree.

    Inputs:
    node :: TreeNode
        the current node being processed
    softmax_values :: the array of softmax values for an example
    """
    if node.index >= 0:
        node.softmax = softmax_values[node.index]
    else:
        for child in node.children:
            update_softmax(child, softmax_values)

def update_priors(node, positive_priors, negative_priors):
    """
    Recursively updates the positive and negative prior values at terminal nodes in a tree.

    Inputs:
    node :: TreeNode
        the current node being processed
    positive_priors :: the array of positive priors
    negative_priors :: the array of negative priors
    """
    if node.index >= 0:
        node.positive_prior = positive_priors[node.index]
        node.negative_prior = negative_priors[node.index]

        for ancestor in node.ancestors:
            ancestor.positive_prior += positive_priors[node.index]
            if ancestor.label != 'unknown':
                ancestor.negative_prior = 1 - ancestor.positive_prior
    else:
        for child in node.children:
            update_priors(child, positive_priors, negative_priors)

def lowest_common_subsumer(terminal_a, terminal_b):
    a_path = list(reversed(terminal_a.ancestors))
    b_path = terminal_b.ancestors

    for ancestor in a_path:
        if ancestor in b_path:
            return ancestor

def load_tree_from_file(file_path, nbins=0):
    tree = GeneralTree()

    with open(file_path, 'r') as f:
        # Read all lines of the file
        all_lines = f.readlines()

    # Remove the MAX_SIZE line and first # line
    all_lines = all_lines[2:] 

    node_label = ''
    node_index = None
    node_parent = None
    node_children = None
    while 'unknown' not in node_label:
        # Read in TreeNode information
        node_label = all_lines.pop(0).rstrip('\n').split()[1]
        node_index = int(all_lines.pop(0).rstrip('\n').split()[1])
        node_parent = all_lines.pop(0).rstrip('\n').split()[1]
        node_children = all_lines.pop(0).rstrip('\n').split()[1:]

        # Set children to empty list if a terminal node
        if node_children[0] == 'None':
            node_children = []

        # Throw away # line
        all_lines.pop(0)

        # Add TreeNodes to tree object if it doesn't already exist
        if not hasattr(tree, node_label):
            setattr(tree, node_label, TreeNode(node_label, node_index, nbins=nbins))
        
        # Retrieve the current node
        node = getattr(tree, node_label)

        # If the parent TreeNode exists, establish connection
        # If the parent doesn't exist, create it then establish connection
        if hasattr(tree, node_parent):
            parent = getattr(tree, node_parent)
            node.set_parent(parent)
        elif node_label != 'unknown':
            setattr(tree, node_parent, TreeNode(node_parent, -1, nbins=nbins))
            parent = getattr(tree, node_parent)
            node.set_parent(parent)

    return tree