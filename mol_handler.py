
def get_HbondIdx(mol, p_index, matches):
    for bond in mol.GetAtomWithIdx(matches[p_index]).GetBonds():
        if bond.GetEndAtom().GetSymbol() == 'H':
            return mol.GetBondBetweenAtoms(matches[p_index],bond.GetEndAtom().GetIdx()).GetIdx()

#ベースになるものとして, Glyでないくっつ
def mol_with_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

