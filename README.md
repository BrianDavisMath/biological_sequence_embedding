# Biological Sequence Embedding
This repo is for sharing the featurization methods developed at the Markey Cancer Center with Sally Ellingson.

## Reference proteins
Reference proteins (proteins encoded by human chromosomes) are sourced from [here](https://www.uniprot.org/proteomes/UP000005640).

The biological sequence associated with the proteins is the amino acid sequence, found in the FASTA. We define the sequence-weight of a protein to be the optimal alignment score of the sequence with itself using [Striped Smith-Waterman](http://scikit-bio.org/docs/0.1.1/core.ssw.html) with [PAM250](https://biopython.org/DIST/docs/api/Bio.SubsMat.MatrixInfo-module.html#pam250) substitution values. Since FASTA may contain extra characters "U" and "O", we assign a value of 4 to those characters, which aren't accounted for in PAM250. We define the similarity of two proteins to be the alignment score divided by the geometric mean of the sequence-weights.

## Reference Ligands
Reference ligands (all metabolites found in
the human body) are sourced from [here](http://www.hmdb.ca/downloads).

The sequence we use for ligands is the binary [fingerprint](https://www.rdkit.org/docs/GettingStartedInPython.html). We use a radius of 8 and a bit-length of 4096 by default, as at smaller radii, molecules with distinct SMILES sequences may have the same binary fingerprint. We include code for reducing fingerprint datasets with high redundancy using a tree data structure.

We use [Tanimoto similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html) as the similarity between two ligands.

## Reference complexes
We consider "complex" space to consist of all pairs from these data sets (although clearly most such pairs do not form complexes).


## Usage
Written for Python 3.6.8 and with [these packages](https://github.com/BrianDavisMath/biological_sequence_embedding/blob/master/packages.txt).

The basic approach of the project is to compute sequence similarity, save the indices and similarities of the 200 most similar sequences to each sequence, then to pass this sparse matrix to UMAP for embedding, or to create a Networkx graph with similarity-weighted edges and embed that.  

#### Protein workflow
After downloading the proteins file we run the following sequence of commands to build our dataset:

  ```
  python extract_reference_fastas.py
  python sparse_distances.py reference_FASTAS.csv protein
  ```
The result will be a .joblib file containing the sparse distance matrix whose entries correspond to the 200 "closest" proteins for each protein. If you want the "symmetric" version, use "protein_sym" as the last argument
#### Ligand workflow
After downloading the ligands file we run the following sequence of commands to build our reference ligand dataset:

  ```
  python extract_reference_smiles.py
  python remove_fingerprint_redundancy.py reference_SMILES.csv 8 4096
  python sparse_distances.py reference_fingerprints_8_4096.csv ligand
  ```
The result will be a .joblib file containing the sparse distance matrix whose entries correspond to the 200 "closest" ligands for each ligand.
The argument "8" is the radius used in the calculation of the fingerprint. Typical values are 4 (resulting in high redundancy), 6, and 8.
The argument 4096 is the bit-length used, a high bit-length reduces the frequency of collisions, again leading fewer redundant fingerprints.

### UMAP embedding
Once you have produced the sparse distance matrix for your protein / ligand data, you can run a grid search for optimal embedding hyperparameters using UMAP with the following command:

  ```
  python sparse_umap_gridsearch.py sparse_distance_mat_{sequence_info}.joblib
  ```
  The script will produce the file ```{sequence_info}_neighbor_metrics.csv```, which you can then inspect. The statistics it reports are for the neighbor_metric of the ```molecule_similarities.py``` module. It is a per-example answer to the question: "What proportion of the closest 200 data points in the embedding where among the 200 most similar sequences?" You might choose UMAP parameters which maximize the 50% (approx. median) value, or the mean value, for example. 
  
  You can then run ```sparse_umap_embedding.py``` with your chosen parameters to get your embedding:
  for example, for UMAP parameters ```n_neighbors=10, min_dist=0.75, dim=2, random_state=42```, you can run
  ```
  python sparse_umap_embedding.py sparse_distance_mat_200_ligand_4_1024.joblib 10, 0.75, 2, 42
  ```
  to produce the file ```200_ligand_4_1024_10_75_2_42_embedding.csv```
  which, when plotted with pyplot:
  ```
  import matplotlib.pyplot as plt
  import pandas as pd
  embedding = pd.read_csv("200_ligand_4_1024_10_75_2_42_embedding.csv", index_col=0, names=['x', 'y'])
  plt.scatter(embedding.x, embedding.y, alpha=0.01)
  ```
  produces the image
  
  ![alt text](https://github.com/BrianDavisMath/biological_sequence_embedding/blob/master/200_ligand_4_1024_10_75_2_4.png "embedded ligands visualization")
  
  

