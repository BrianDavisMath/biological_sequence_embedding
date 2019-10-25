import pandas as pd
import re

get_id = re.compile("<DATABASE_ID>\n[\dA-Z]+", re.IGNORECASE)
get_smiles = re.compile("<SMILES>\n[.%#+\-\dA-Z()\[\]=@/\\\]+\n", re.IGNORECASE)


smiles = {}
fail_count = 0
file = "structures.sdf"
with open(file, "r") as smiles_file:
    text = smiles_file.read()
mols = iter(text.split("$$$$"))
for mol in mols:
    id_ = get_id.search(mol)
    smiles_ = get_smiles.search(mol)
    if bool(id_) and bool(smiles_):
        smiles[id_.group()[14:]] = smiles_.group()[9:-1]
    else:
        fail_count += 1
print(f"{fail_count} mol(s) failed to produce smiles.")
pd.Series(smiles).to_csv("reference_SMILES.csv", header=False)
