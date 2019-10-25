import pandas as pd
import joblib
from zipfile import ZipFile
import numpy as np
from umap import UMAP
import sys
from sklearn.neighbors import NearestNeighbors
import itertools
import warnings
import multiprocessing as mp
warnings.filterwarnings('ignore')
sys.setrecursionlimit(10**6)


nhbrs = [2, 3, 5, 10, 25]
dsts = [0.0, 0.01, 0.1, 0.5]
dims = [2, 5, 10, 100]
rands = [42, 21, 7, 3]
params = list(itertools.product(*[nhbrs, dsts, dims, rands]))


def main(sparse_distances_file_):
    # load similarities
    print("Loading similarities...")
    file_name, extension = sparse_distances_file_.split(".")
    if extension == "zip":
        with ZipFile(file_name + ".zip") as myzip:
            with myzip.open(file_name + ".joblib") as f:
                similarities = joblib.load(f)
    if extension == "joblib":
        similarities = joblib.load(file_name + ".joblib")

    # extract similarity data
    print("Building nearest neighbors model...")
    tmp_mat = similarities.tocoo()
    rows = tmp_mat.row
    max_neighbors = len(rows) - np.count_nonzero(rows)
    size = int(len(rows) / max_neighbors)
    highest_sims = tmp_mat.col.reshape((size, max_neighbors)).tolist()
    nn = NearestNeighbors(n_neighbors=max_neighbors)

    def neighbor_score(params_):
        num_neighbors, min_dist, dim, rand_state = params_
        neighbor_performance = dict({"num_neighbors": num_neighbors,
                                     "min_dist": min_dist,
                                     "dim": dim,
                                     "rand_state": rand_state})
        try:
            embedding = UMAP(n_neighbors=num_neighbors,
                             min_dist=min_dist,
                             n_components=dim,
                             random_state=rand_state
                             ).fit_transform(similarities)
            nn.fit(embedding)
            nearest_embed = nn.kneighbors(embedding, return_distance=False).tolist()
            counts = np.array([len(set(a).intersection(set(b))) for a, b in zip(nearest_embed, highest_sims)]) / max_neighbors
            neighbor_performance.update(pd.Series(counts).describe().to_dict())
            print(neighbor_performance)
            return neighbor_performance
        except Exception:
            print(f"Embedding failed with parameters: {neighbor_performance}")
            pass

    print("Performing parameter gridsearch...")
    performances = mp.Pool(processes=15).imap_unordered(neighbor_score, params)
    pd.DataFrame(performances).to_csv("neighbor_metrics.csv")
    print("Done- saving metrics to neighbor_metrics.csv")


if __name__ == "__main__":
    _, sparse_distances_file = sys.argv
    main(sparse_distances_file)






