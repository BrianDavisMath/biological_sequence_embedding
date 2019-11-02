import pandas as pd
import joblib
from zipfile import ZipFile
import numpy as np
from molecule_similarities import get_neighbor_score_func
import sys
import itertools
import warnings
from pathos.multiprocessing import ProcessingPool as Pool
warnings.filterwarnings('ignore')
sys.setrecursionlimit(10**6)


nhbrs = [2, 3, 5, 10, 25]
dsts = [0.0, 0.01, 0.1, 0.5]
dims = [2, 5, 10, 100]
rands = [42, 21, 7, 3]

feed = list(itertools.product(*[nhbrs, dsts, dims, rands]))


if __name__ == "__main__":
    _, sparse_distances_file = sys.argv
    # load similarities
    print("Loading similarities...")
    file_name, extension = sparse_distances_file.split(".")
    if extension == "zip":
        with ZipFile(file_name + ".zip") as myzip:
            with myzip.open(file_name + ".joblib") as f:
                similarities = joblib.load(f)
    if extension == "joblib":
        similarities = joblib.load(sparse_distances_file)
    # extract similarity data
    print("Building nearest neighbors model...")
    tmp_mat = similarities.tocoo()
    rows = tmp_mat.row
    max_neighbors = len(rows) - np.count_nonzero(rows)
    size = int(len(rows) / max_neighbors)
    highest_sims = tmp_mat.col.reshape((size, max_neighbors)).tolist()
    func = get_neighbor_score_func(max_neighbors, similarities, highest_sims)
    print("Running grid search...")
    pool = Pool()
    performances = pool.map(func, feed)
    pd.DataFrame(performances).to_csv(file_name[20:] + "_neighbor_metrics.csv")
    print(f"Done- saving metrics to {file_name[20:]}_neighbor_metrics.csv")
