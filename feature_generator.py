from mordred import Calculator, descriptors
import numpy as np
from rdkit.Chem import AllChem

from smiles_handler import calc_smiles_skip_connection


def calc_MorganCount(mol, r = 2, dimension = 2048):
    info = {}
    _fp = AllChem.GetMorganFingerprint(mol, r, bitInfo=info)
    count_list = [0] * dimension
    for key in info:
        pos = key % dimension
        count_list[pos] += len(info[key])
    return count_list


def calc_feature_skip_connection(smi, peptide_feature, skip, feature, descriptor_dimension = 2048):
    vertical_list = calc_smiles_skip_connection(smi, peptide_feature, skip = 4)
  
 
    vertical_feature_list = []
    for mol in vertical_list:
        if feature == 'Morgan_r2':
            vertical_feature_list.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, descriptor_dimension))
        elif feature == 'Morgan_r4':
            vertical_feature_list.append(AllChem.GetMorganFingerprintAsBitVect(mol, 4, descriptor_dimension))
        elif feature == 'MACCS':
            vertical_feature_list.append(AllChem.GetMACCSKeysFingerprint(mol))
        elif feature == 'Morgan_r2_count':
            vertical_feature_list.append(calc_MorganCount(mol, 2, descriptor_dimension))
        elif feature == 'Morgan_r4_count':
            vertical_feature_list.append(calc_MorganCount(mol, 4, descriptor_dimension))
    vertical_feature = np.mean(vertical_feature_list, axis = 0)
    return vertical_feature


def calc_mordred_descriptor(mol_list):
    calc = Calculator(descriptors, ignore_3D = True)
    mordred_df = calc.pandas(mol_list)
    df_descriptors = mordred_df.astype(str)
    masks = df_descriptors.apply(lambda d: d.str.contains('[a-zA-Z]' ,na=False))
    df_descriptors = df_descriptors[~masks]
    df_descriptors = df_descriptors.astype(float)
    modred_descriptor = df_descriptors.dropna(axis=1, how='any')

    return modred_descriptor

