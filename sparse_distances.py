import numpy as np
import pandas as pd
from molecule_similarities import DataSet
from scipy.sparse import csr_matrix
import joblib
import sys
sys.setrecursionlimit(10**6)


num_neighbors = 200


def main(sequence_file, sequence_type):
    sequences = pd.read_csv(sequence_file, header=-1).set_index(0).values.flatten()
    size = len(sequences)
    dataset = DataSet(sequences, sequence_type, n_nhbrs=num_neighbors)
    output_file = sequence_file.split(".")[0] + f"_sparse_distance_mat_{sequence_type}_{num_neighbors}.joblib"
    nhbrs, dsts = dataset.neighbors(sequences)
    # embed reference_data by building sparse distance matrix
    rows = np.sort(list(range(size)) * num_neighbors)
    cols = nhbrs[:, :num_neighbors].flatten()
    data = dsts[:, :num_neighbors].flatten()
    sparse_dist_mat = csr_matrix((data, (rows, cols)), shape=(size, size))
    joblib.dump(sparse_dist_mat, output_file)
    print(f"Done building sparse distance matrix. Saving to: \n {output_file}")


if __name__ == "__main__":
    _, sequence_file, sequence_type = sys.argv
    main(sequence_file, sequence_type)

"""Example usage:
python reference_embedding.py reference_FASTAS.csv protein_sym"""