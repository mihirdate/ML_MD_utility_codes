import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, ReducedGraphs
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.EState import Fingerprinter
from rdkit.Chem import Descriptors

def fingerprint(mol, ftype="MACCSKeys", radius=2, bits=1024):
    npfp = np.zeros((1,))
    if ftype == "MACCSKeys":
        DataStructs.ConvertToNumpyArray(AllChem.GetMACCSKeysFingerprint(mol), npfp)
    elif ftype == "Avalon":
        DataStructs.ConvertToNumpyArray(GetAvalonFP(mol), npfp)
    elif ftype == "ECFP":
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius, bits), npfp)
    elif ftype == "ErG":
        npfp = ReducedGraphs.GetErGFingerprint(mol)
    elif ftype == "Estate":
        npfp = Fingerprinter.FingerprintMol(mol)[0]
    else:
        raise TypeError()
    return npfp

def descriptors(mol):
    calc=MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    ds = np.asarray(calc.CalcDescriptors(mol))
    return ds

def calculate_rdk_features(file_in, file_out):
    dfRaw_experimental = pd.read_csv(file_in)
    smiles = dfRaw_experimental['SMILES']
    molecule_name = dfRaw_experimental['Molecule Name']
    print("Reading", len(smiles), "PROTACS SMILES")

    # Calculate fingerprints and structural features
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    dfRDKitFeature = pd.DataFrame([descriptors(mol) for mol in mols])
    dfdescriptors_list = pd.DataFrame([x[0] for x in Descriptors._descList])

    dfRDKitFeature = dfRDKitFeature.rename(columns=dfdescriptors_list[0])
    dfRDKitFeature = pd.concat([molecule_name,smiles,dfRDKitFeature], axis = 1)
    print("The shape of RDK feature matrix is ", dfRDKitFeature.shape)

    dfRDKitFeature.to_csv(file_out, sep=',')

