## Deep learning and array processing libraries
import numpy as np 
import pandas as pd 

## Tree-Related Imports
from anytree import NodeMixin, RenderTree

##### Classes for a general tree and for an individual tree node

class GeneralTree(object):
    """
    Container class for a general tree used in the approach
    """
    def __init__(self):
        pass
    
    def __repr__(self):
        return str(RenderTree(self.unknown))

class TreeNode(NodeMixin):
    """
    Class for each node in a GeneralTree
    """
    def __init__(self, label, index, softmax=0, nbins=0, parent=None):
        super(TreeNode, self).__init__()
        self.label = label
        self.index = index
        self.softmax = softmax
        self.positive_prior = 0
        self.negative_prior = 0
        self.positive_hist = np.zeros(nbins, dtype=np.float32)
        self.negative_hist = np.zeros(nbins, dtype=np.float32)
        self.posteriors = np.zeros(nbins, dtype=np.float32)
        self.parent = parent

    def set_parent(self, parent):
        """
        Assign the parent node for a given node
        """
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
        return f'[TreeNode: {self.label} / {self.index} -- {np.round(self.posteriors, decimals=3)}]'

##### Some other utility functions relating to the hierarchy

def update_softmax(node, softmax_values):
    """
    Recursively updates the softmax values at terminal nodes in a tree
    """
    if node.index >= 0:
        node.softmax = softmax_values[node.index]
    else:
        for child in node.children:
            update_softmax(child, softmax_values)

def update_priors(node, positive_priors, negative_priors):
    """
    Recursively updates the positive and negative prior values at terminal nodes in a tree
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
    """
    Determine the lowest common subsumer between two terminal nodes in a tree
    """
    a_path = list(reversed(terminal_a.ancestors))
    b_path = terminal_b.ancestors

    for ancestor in a_path:
        if ancestor in b_path:
            return ancestor

def load_tree_from_file(file_path, nbins=0):
    """
    Create a GeneralTree object and populate it based on a txt file
    """
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