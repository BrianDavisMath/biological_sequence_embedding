import pandas as pd
import joblib
from zipfile import ZipFile
from umap import UMAP
import sys
import warnings
warnings.filterwarnings('ignore')
sys.setrecursionlimit(10**6)


if __name__ == "__main__":
    _, sparse_distances_file, num_neighbors, min_dist, dim, rand_state = sys.argv
    # load similarities
    print("Loading similarities...")
    file_name, extension = sparse_distances_file.split(".")
    if extension == "zip":
        with ZipFile(file_name + ".zip") as myzip:
            with myzip.open(file_name + ".joblib") as f:
                similarities = joblib.load(f)
    if extension == "joblib":
        similarities = joblib.load(sparse_distances_file)
    embedding = UMAP(n_neighbors=int(num_neighbors),
                     min_dist=float(min_dist),
                     n_components=int(dim),
                     random_state=int(rand_state)
                     ).fit_transform(similarities)
    pd.DataFrame(embedding).to_csv(file_name.split(".")[0][20:]\
                                   + f"_{num_neighbors}_" + str(min_dist).replace(".", "")\
                                   + f"_{dim}_{rand_state}_embedding.csv")
