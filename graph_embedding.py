import pandas as pd
import joblib
from zipfile import ZipFile
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import sys
from sklearn.neighbors import NearestNeighbors
import itertools
import warnings
import multiprocessing as mp
warnings.filterwarnings('ignore')
sys.setrecursionlimit(10**6)

"""nhbrs = [2, 3, 5, 10, 25]
dsts = [0.0, 0.01, 0.1, 0.5]
dims = [2, 5, 10, 100]
rands = [42, 21, 7, 3]"""
nhbrs = [2]
dsts = [0.0]
dims = [2]
rands = [42]


#read graph
reference_graph = nx.from_scipy_sparse_matrix(similarities)

# set parameters
walk_length_ = 10
num_walks_ = 80
p_ = 0.25
q_ = 4
window_size_ = 5
iter_ = 3

#init model
model = Node2Vec(reference_graph, walk_length=walk_length_,
                 num_walks=num_walks_, p=p_, q=q_, workers=15)

# train model
model.train(window_size=window_size_, iter=iter_)

# get embedding vectors
embedding = model.get_embeddings()


def neighbor_score(num_neighbors_, min_dist_, dim_, rand_state_,
                   max_neighbors_, similarities_, highest_sims_):
    neighbor_performance = dict({"num_neighbors": num_neighbors_,
                                 "min_dist": min_dist_,
                                 "dim": dim_,
                                 "rand_state": rand_state_})
    try:
        embedding = UMAP(n_neighbors=num_neighbors_,
                         min_dist=min_dist_,
                         n_components=dim_,
                         random_state=rand_state_
                         ).fit_transform(similarities_)
        nn = NearestNeighbors(n_neighbors=max_neighbors)
        nn.fit(embedding)
        nearest_embed = nn.kneighbors(embedding, return_distance=False).tolist()
        counts = [len(set(a).intersection(set(b))) for a, b in zip(nearest_embed, highest_sims_)]
        counts = np.array(counts) / max_neighbors_
        neighbor_performance.update(pd.Series(counts).describe().to_dict())
        print(neighbor_performance)
        return neighbor_performance
    except Exception:
        print(f"Embedding failed with parameters: {neighbor_performance}")
        pass


def get_neighbor_score_func(max_neighbors_, similarities_, highest_sims_):
    def neighbor_score_func(params_):
        num_neighbors, min_dist, dim, rand_state = params_
        return neighbor_score(num_neighbors, min_dist, dim, rand_state,
                              max_neighbors_, similarities_, highest_sims_)
    return neighbor_score_func


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
        similarities = joblib.load(file_name + ".joblib")
    # extract similarity data
    print("Building nearest neighbors model...")
    tmp_mat = similarities.tocoo()
    rows = tmp_mat.row
    max_neighbors = len(rows) - np.count_nonzero(rows)
    size = int(len(rows) / max_neighbors)
    highest_sims = tmp_mat.col.reshape((size, max_neighbors)).tolist()
    print("Performing parameter grid search...")
    feed = list(itertools.product(*[nhbrs, dsts, dims, rands]))
    func = get_neighbor_score_func(max_neighbors, similarities, highest_sims)
    performances = mp.Pool(processes=15).map(func, feed)
    pd.DataFrame(performances).to_csv(file_name + "_neighbor_metrics.csv")
    print("Done- saving metrics to neighbor_metrics.csv")
