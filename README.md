# Biological Sequence Embedding
This repo is for sharing the featurization methods developed at the Markey Cancer Center with Sally Ellingson.

## Reference proteins
Reference proteins (proteins encoded by human chromosomes) are sourced from [here](https://www.uniprot.org/proteomes/UP000005640). The biological sequence associated with the proteins is the amino acid sequence, found in the FASTA. We define the sequence-weight of a protein to be the optimal alignment score of the sequence with itself using [Striped Smith-Waterman](http://scikit-bio.org/docs/0.1.1/core.ssw.html) with [PAM250](https://biopython.org/DIST/docs/api/Bio.SubsMat.MatrixInfo-module.html#pam250) substitution values. Since FASTA may contain extra characters "U" and "O", we assign a value of 4 to those characters, which arent accounted for in PAM250.

We define the similarity of two proteins to be the alignment score divided by the geometric mean of the sequence-weights.

## Reference Ligands
Reference ligands (all metabolites found in
the human body) are sourced from [here](http://www.hmdb.ca/downloads).

We consider ``complex" space to consist of all pairs from these data sets.
