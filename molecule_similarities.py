import numpy as np
from skbio.alignment import StripedSmithWaterman
from Bio.SubsMat import MatrixInfo
from sklearn.neighbors import NearestNeighbors
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances
import multiprocessing as mp
import warnings


def get_fingerprints(smiles_string):
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 8, nBits=4096).GetOnBits())
    except:
        return np.nan


def neighbor_score(nearest_viz_neighbors, latent_neighbors, n_neighbors):
    counts = [len(set(a).intersection(set(b))) for a, b in zip(nearest_viz_neighbors, latent_neighbors)]
    return np.array(counts) / n_neighbors


def subs_mat(subs):
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
                   'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z', 'X']
    a_a_pairs = {}
    for c in amino_acids:
        a_a_pairs[c] = {'*': 0}
        for s in amino_acids:
            try:
                a_a_pairs[c][s] = subs[(c, s)]
            except KeyError:
                a_a_pairs[c][s] = subs[(s, c)]
    dummy_dict = {'*': {'*': 0}}
    for c in amino_acids:
        dummy_dict['*'][c] = 0
    a_a_pairs.update(dummy_dict)
    return a_a_pairs


substitution_data = subs_mat(MatrixInfo.pam250)


def seq_norm(seq_):
    weight = 0
    for char in seq_:
        try:
            weight += substitution_data[char][char]
        except KeyError:
            weight += 4
            if char not in ["U", "O"]:
                warnings.warn(f"(Brian) Sequence contained unrecognized residue: '{char}'."
                              "Normalizing using default subs value of 4.")
    if weight == 0:
        raise NameError("(Brian) Empty sequences are bad, m'kay?")
    return weight


def _fasta_similarity_func(seq):
    norm = seq_norm(seq)
    query = StripedSmithWaterman(seq, protein=True,
                                 substitution_matrix=substitution_data)

    def similarity(_seq_):
        norm_ = seq_norm(_seq_)
        return query(_seq_).optimal_alignment_score / norm
    return np.vectorize(similarity)


def _fasta_similarity_func_sym(seq):
    norm = seq_norm(seq)
    query = StripedSmithWaterman(seq, protein=True,
                                 substitution_matrix=substitution_data)

    def similarity(_seq_):
        norm_ = seq_norm(_seq_)
        return query(_seq_).optimal_alignment_score / np.sqrt(norm * norm_)
    return np.vectorize(similarity)


def _fingerprint_similarity_func(seq_array):
    seq_array = seq_array.reshape(1, -1).astype(bool)

    def func_(seqs_):
        return 1 - pairwise_distances(seq_array.astype(bool), seqs_, metric="jaccard")
    return func_


def _get_similarity_func(mol_type):
    if mol_type == "protein":
        return _fasta_similarity_func
    if mol_type == "protein_sym":
        return _fasta_similarity_func_sym
    if mol_type == "ligand":
        return _fingerprint_similarity_func
    else:
        raise ValueError("Valid molecule types are 'ligand', 'protein;, and 'protein_sym'.")


def filter_distances(sims, n_sims):
    neighbors = np.argsort(sims)[:n_sims]
    return neighbors.tolist(), sims[neighbors].flatten().tolist()


class DataSet:
    def __init__(self, internal_sequences, mol_type, viz_embedding=None, n_nhbrs=200):
        self.n_neighbors = n_nhbrs
        self.mol_type = mol_type
        self.internal_sequences = internal_sequences
        self.nn_ = NearestNeighbors(n_neighbors=n_nhbrs)
        self.viz = viz_embedding
        if viz_embedding is not None:
            self.nn_.fit(viz_embedding)

    def nn(self, vectors):
        if len(np.shape(vectors)) == 1:
            vectors = vectors.reshape((1, -1))
        return self.nn_.kneighbors(vectors, return_distance=False)

    def update_nn(self):
        self.nn_.fit(self.viz)

    def get_similarities(self, internal_sequence):
        similarity_func = _get_similarity_func(self.mol_type)
        try:
            mol_similarities = similarity_func(internal_sequence)
        except ValueError:
            print("similarity_func failed for this molecule. Check mol_type?")
        return mol_similarities(self.internal_sequences)

    def get_neighbors(self, external_sequence):
        all_dists = 1 - self.get_similarities(external_sequence).flatten()
        neighbors, distances = filter_distances(all_dists, self.n_neighbors)
        return neighbors, distances

    def neighbors(self, external_sequences):
        with mp.Pool(processes=(max(1, mp.cpu_count() - 1))) as p:
            tmp = np.array(p.map(self.get_neighbors, external_sequences, chunksize=10))
        return tmp[:, 0, :].astype(int), tmp[:, 1, :]

    def initial_embedding(self, ref_neighbors, ref_dsts):
        relevant_coords = np.take(self.viz, ref_neighbors, axis=0)
        sims_ = 1 - ref_dsts ** 2
        sims_ /= np.sum(sims_, axis=1).reshape(-1, 1)
        tmp = [np.array(a).dot(b).tolist() for a, b in zip(sims_.tolist(), relevant_coords.tolist())]
        return np.array(tmp)
