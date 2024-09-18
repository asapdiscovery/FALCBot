from rdkit import Chem

def _is_valid_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    else:
        return True


def _rdkit_smiles_roundtrip(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol)
