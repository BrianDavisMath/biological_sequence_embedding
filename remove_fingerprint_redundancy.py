import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances
import sys
from scipy.sparse import dok_matrix
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')
sys.setrecursionlimit(10**6)

fingerprint_radius = 8
fingerprint_bits = 4096
smiles_filename = "reference_SMILES.csv"


def get_fingerprints(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        on_bits = AllChem.GetMorganFingerprintAsBitVect(mol, fingerprint_radius, nBits=fingerprint_bits).GetOnBits()
        return np.array(on_bits)
    except Exception:
        return np.nan


class Node:
    def __init__(self, content, parent, children, ids):
        self.content = content
        self.parent = parent
        self.children = children
        if len(content) == 0:
            self.first_char = None
        else:
            self.first_char = content[0]
        self.ids = ids

    def accum_fingerprint(self):
        if self.parent is None:
            return []
        else:
            return self.parent.accum_fingerprint() + list(self.content)


root = Node(content=np.array([]), parent=None, children=[], ids=[])
tree = [root]


def insert(name, word, node):
    node_word = node.content
    run = min(len(node_word), len(word))
    for i in range(run):
        node_char = node_word[i]
        char = word[i]
        if not (char == node_char):
            # word differs from node content somewhere - need to split content
            # create new ''middle'' node
            middle_node = Node(content=node_word[:i], parent=node.parent, children=[node], ids=[])
            tree.append(middle_node)
            # create new node for id
            new_leaf = Node(content=word[i:], parent=middle_node, children=[], ids=[name])
            tree.append(new_leaf)
            # update node parent
            node.parent.children.remove(node)
            node.parent.children.append(middle_node)
            # update node
            node.content = node_word[i:]
            node.parent = middle_node
            node.first_char = node_word[i]
            # update middle_node
            middle_node.children.append(new_leaf)
            return
    # if we reach this line, then the node contents match the word until one is exhausted
    # if both are exhausted, then the fingerprints are identical-- add id to node.ids
    if len(node_word) == len(word):
        node.ids.append(name)
        return
    # if node_word alone is exhausted, either add new leaf for excess word characters or call insert on a node child
    if len(node_word) == run:
        char_list = [child.first_char for child in node.children]
        if word[run] not in char_list:
            new_leaf = Node(content=word[run:], parent=node, children=[], ids=[name])
            tree.append(new_leaf)
            node.children.append(new_leaf)
            return
        else:
            insert(name, word[run:], node.children[char_list.index(word[run])])
            return
    # if word alone is exhausted, insert is as a new node with node as a child
    else:
        # create new ''middle'' node
        middle_node = Node(word, parent=node.parent, children=[node], ids=[name])
        tree.append(middle_node)
        # update node
        node.content = node_word[run:]
        node.parent = middle_node
        node.first_char = node_word[run]
        return


# read in smiles
smiles = pd.read_csv(smiles_filename, names=["id", "SMILES"]).set_index("id")
# compute fingerprints
fingerprints = smiles.SMILES.apply(get_fingerprints).dropna()
# add to tree
for i in range(len(fingerprints)):
    insert(fingerprints.index[i], fingerprints.iloc[i], root)
# gather unique fingerprints and form dictionary of ids
relevant_nodes = [node for node in tree if node.ids != []]
prints = {}
ids = {}
for node_index, node in enumerate(relevant_nodes):
    fingerprint = node.accum_fingerprint()
    binary_fingerprint = np.zeros(fingerprint_bits)
    for index in fingerprint:
        binary_fingerprint[index] = 1
    prints[node_index] = binary_fingerprint.astype(bool)
    for mol_id in node.ids:
        ids[mol_id] = node_index
prints_array = pd.DataFrame(prints).transpose().values
max_neighbors = 200