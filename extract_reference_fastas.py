import pandas as pd


fastas = {}
with open("UP000005640_9606.fasta", "r") as fastas_file:
    text = fastas_file.read()
prots = iter(text.split(">")[1:])
for prot in prots:
    try:
        ID = prot.split(" ")[0]
        FASTA = prot[prot.index("\n"):].replace("\n", "")
        fastas[ID] = FASTA
    except ValueError:
        continue

pd.Series(fastas).to_csv("reference_FASTAS.csv", header=None)
