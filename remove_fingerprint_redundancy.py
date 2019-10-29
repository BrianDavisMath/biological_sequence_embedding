import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
import sys

warnings.filterwarnings('ignore')


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


def main(smiles_file_, fingerprint_radius_, fingerprint_bits_):
    # read in smiles
    smiles = pd.read_csv(smiles_file_, names=["id", "SMILES"]).set_index("id")
    # compute fingerprints
    print("Computing fingerprints")
    fingerprints = smiles.SMILES.apply(get_fingerprints).dropna()
    # add to tree
    print("Inserting fingerprints into search tree.")
    for i in range(len(fingerprints)):
        insert(fingerprints.index[i], fingerprints.iloc[i], root)
    # gather unique fingerprints and form dictionary of ids
    relevant_nodes = [node for node in tree if node.ids != []]
    prints = {}
    ids = {}
    for node_index, node in enumerate(relevant_nodes):
        fingerprint = node.accum_fingerprint()
        binary_fingerprint = np.zeros(fingerprint_bits_)
        for index in fingerprint:
            binary_fingerprint[index] = 1
        prints[node_index] = binary_fingerprint.astype(bool)
        for mol_id in node.ids:
            ids[mol_id] = node_index
    print(f"Finished finding unique fingerprints. Saving at 'reference_fingerprints_{fingerprint_radius_}_{fingerprint_bits_}.joblib'")
    prints_array = pd.DataFrame(prints).transpose().values
    joblib.dump(prints_array, f"reference_fingerprints_{fingerprint_radius_}_{fingerprint_bits_}.joblib")


if __name__ == "__main__":
    _, smiles_file, fingerprint_radius, fingerprint_bits = sys.argv

    def get_fingerprints(smiles_string):
        try:
            mol = Chem.MolFromSmiles(smiles_string)
            on_bits = AllChem.GetMorganFingerprintAsBitVect(mol, fingerprint_radius, nBits=fingerprint_bits).GetOnBits()
            return np.array(on_bits)
        except Exception:
            return np.nan

    main(smiles_file, fingerprint_radius, fingerprint_bits)
