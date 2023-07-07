from rdkit import Chem
from rdkit.Chem import AllChem

from mol_handler import get_HbondIdx


def calc_smiles_skip_connection(smi, peptide_feature, skip=4):
    pep_len = len([v for v in  peptide_feature[4:] if v >= 0])
    mol = Chem.MolFromSmiles(smi)
    linker_count = 0

    while mol.HasSubstructMatch(Chem.MolFromSmarts('[O][C:1]@[C:2]=[C:3]@[C:4][O]')): 
        bis = mol.GetSubstructMatches(Chem.MolFromSmarts('[O][C:1]@[C:2]=[C:3]@[C:4][O]'))
        for bs in bis:
            bs = [mol.GetBondBetweenAtoms(bs[2], bs[3]).GetIdx()]
            fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=False)
        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True)
        except:
            return None
        mol = fragments[0]
        linker_count += 1

    MC_smiles = Chem.MolFromSmiles('NC(=O)C' * (pep_len))
    matches = mol.GetSubstructMatches(MC_smiles)[0]
    vertical_list = [Chem.MolFromSmiles('[H][1*]')] * skip

    for i in range(pep_len):
        skip_base = i % skip
        
        if i < pep_len - skip:
            bs = [mol.GetBondBetweenAtoms(matches[i*4 + 1], matches[i*4 + 3]).GetIdx(), 
                  mol.GetBondBetweenAtoms(matches[i*4 + 3], matches[i*4 + 4]).GetIdx()]
            fragments_mol = Chem.FragmentOnBonds(mol, bs, 
                                                 addDummies=True, 
                                                 dummyLabels=[(int(i/skip + 1), int(i/skip + 1)), (int(i/skip + 2), int(i/skip + 2))])
            fragments = Chem.GetMolFrags(fragments_mol,asMols=True)
            for fragment in fragments:
              if '[' + str(int(i/skip + 1)) + '*]' in Chem.MolToSmiles(fragment) and '[' + str(int(i/skip + 2)) + '*]' in Chem.MolToSmiles(fragment):
                aa_fragment = fragment
                break
            reaction_pattern = '[*:1][' + str(int(i/skip + 1)) + '*].[' + str(int(i/skip + 1)) + '*][*:2] >> [*:1][*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            vertical_list[skip_base] = rxn.RunReactants([vertical_list[skip_base], aa_fragment])[0][0]

        else:
            reaction_pattern = '[*:1][' + str(int(i/skip + 1)) + '*] >> [*:1][H]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            vertical_list[skip_base] = rxn.RunReactants([vertical_list[skip_base]])[0][0]

    vertical_list = [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in vertical_list]
    return vertical_list


def replaceX_smiles(smi, peptide_feature, base_atom ='P'):
    pep_len = len([v for v in  peptide_feature[4:] if v >= 0])
    mol = Chem.MolFromSmiles(smi)
    tmp = Chem.MolFromSmiles('NC(=O)C' * (pep_len))  
    mc_pattern, pc_pattern = '', ''

    for i in range(pep_len):
        mc_pattern += '[N:' + str(i*4 + 1) + '][C:' + str(i*4 + 2) + '](=[O:' + str(i*4 + 3) + '])[C:' + str(i*4 + 4) + ']'
        pc_pattern += '[N:' + str(i*4 + 1) + '][C:' + str(i*4 + 2) + '](=[O:' + str(i*4 + 3) + '])[' + base_atom + ':' + str(i*4 + 4) + ']([H])([H])'

    reaction_pattern = mc_pattern + '>>' + pc_pattern
    rxn = AllChem.ReactionFromSmarts(reaction_pattern)
    x = rxn.RunReactants([mol])[0]
    return Chem.MolToSmiles(x[0])


def calc_graph_connect(smi, peptide_feature, skip=4, base_atom='P'):
    pep_len = len([v for v in  peptide_feature[4:] if v >= 0])
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    pc_pattern = ('NC(=O)' + base_atom) * pep_len
    matches = mol.GetSubstructMatches(Chem.MolFromSmiles(pc_pattern))[0]

    for i in range(pep_len - skip):
        matches = mol.GetSubstructMatches(Chem.MolFromSmiles(pc_pattern))[0]
        bs = [get_HbondIdx(mol, i*4 + 3, matches), get_HbondIdx(mol, i*4 + 3 + skip*4, matches)]
        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1,1), (2,2)])
        fragments = Chem.GetMolFrags(fragments_mol, asMols=True)
        reaction_pattern = "([*1]-[" + base_atom + ":1].[" + base_atom + ":2]-[*2])>>[" + base_atom + ":1]-[" + base_atom + ":2]"
        rxn = AllChem.ReactionFromSmarts(reaction_pattern)
        mol = rxn.RunReactants([fragments[0]])[0][0]

    x = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(x)