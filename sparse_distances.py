import numpy as np
import pandas as pd
from molecule_similarities import DataSet
from scipy.sparse import csr_matrix
import joblib
import sys
sys.setrecursionlimit(10**6)


num_neighbors = 200


def main(sequence_file_, sequence_type_):
    if "protein" in sequence_type_:
        sequences = pd.read_csv(sequence_file_, header=-1).set_index(0).values.flatten()
    elif "ligand" in sequence_type_:
        sequences = joblib.load(sequence_file_)
    size = len(sequences)
    print(f"sequences length: {size}")
    dataset = DataSet(sequences, sequence_type_, n_nhbrs=num_neighbors)
    if "ligand" in sequence_type_:
        sequence_params = sequence_file_.split("fingerprints")[1].split(".")[0]
    else:
        sequence_params = ""
    output_file = f"sparse_distance_mat_{num_neighbors}_" + sequence_type_ + sequence_params + ".joblib"
    nhbrs, dsts = dataset.neighbors(sequences)
    # embed reference_data by building sparse distance matrix
    rows = np.sort(list(range(size)) * num_neighbors)
    cols = nhbrs[:, :num_neighbors].flatten()
    data = dsts[:, :num_neighbors].flatten()
    sparse_dist_mat = csr_matrix((data, (rows, cols)), shape=(size, size))
    joblib.dump(sparse_dist_mat, output_file)
    print(f"Done building sparse distance matrix. Saving to: \n {output_file}")


if __name__ == "__main__":
    *_, sequence_file, sequence_type = sys.argv
    main(sequence_file, sequence_type)

"""Example usage:
python sparse_distances.py reference_fingerprints.joblib ligand"""
