from rdkit import Chem
from rdkit.Chem import AllChem 

def peptide_feature2AA_seq(pf, AA_keys, ct_list, nt_list):
    aa_seq = ''

    for j, k in enumerate(pf[4:]):
        if j in pf[2:4]:
            aa_seq += '='
        aa_seq += AA_keys[k]

    seq = ct_list[pf[0]] + '-' + aa_seq + '-' + nt_list[pf[1]]
    return seq


def make_joint_MC(base_mol, MC_mol, pep_len):
    for i in range(pep_len):
        matches = base_mol.GetSubstructMatches(MC_mol)[0]
        atom_index = matches[i*4 + 1]
        Ca_atom = base_mol.GetAtomWithIdx(atom_index)
        c_beta_idx = [x.GetIdx() for x in Ca_atom.GetNeighbors() if x.GetIdx() not in list(matches)][0]
        atom_pair = [Ca_atom.GetIdx(), c_beta_idx]
        bs = [base_mol.GetBondBetweenAtoms(atom_pair[0], atom_pair[1]).GetIdx()]
        fragments_mol = Chem.FragmentOnBonds(base_mol, bs, addDummies=True, dummyLabels=[(i + 1, i + 1)])

        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True)
        except:
            print("error")

        base_mol = fragments[0]
        matches = base_mol.GetSubstructMatches(MC_mol)[0]
        atom_index = matches[i*4 + 1]
        Ca_atom = base_mol.GetAtomWithIdx(atom_index)
        h_counter = 0
        cb_counter = 0

        for x in Ca_atom.GetNeighbors() :
            if x.GetAtomicNum() == 1:
                h_counter += 1
                h_idx = x.GetIdx()
            elif (x.GetIdx() not in list(matches)) and cb_counter == 0 and x.GetAtomicNum() != 1 and h_counter == 0:
                cb_counter += 1
                h_idx = x.GetIdx()
            else:
                continue

        atom_pair = [Ca_atom.GetIdx(), h_idx]
        bs = [base_mol.GetBondBetweenAtoms(atom_pair[0], atom_pair[1]).GetIdx()]
        fragments_mol = Chem.FragmentOnBonds(base_mol, bs, addDummies=True, dummyLabels=[(i + 1 + 100, i + 1 + 100)])

        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True)
        except:
            print("error")
        
        base_mol = fragments[0]
        matches = base_mol.GetSubstructMatches(MC_mol)[0]
        atom_index = matches[i*4 + 0]
        N_atom = base_mol.GetAtomWithIdx(atom_index)
        h_idx = [x.GetIdx() for x in N_atom.GetNeighbors() if x.GetAtomicNum() == 1][0]
        atom_pair = [N_atom.GetIdx(), h_idx]
        bs = [base_mol.GetBondBetweenAtoms(atom_pair[0], atom_pair[1]).GetIdx()]
        fragments_mol = Chem.FragmentOnBonds(base_mol, bs, addDummies=True, dummyLabels=[(i + 1 + 200, i + 1 + 200)])
        
        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True)
        except:
            print("error")
        
        base_mol = fragments[0]

    base_mol = Chem.RemoveHs(base_mol)
    base_smi = Chem.MolToSmiles(base_mol)
    return base_smi, base_mol


def make_new_peptide(joint_MC_mol, AA_keys, AA_joint, input_aa_list, AA_dict):
    AA_key_list = [AA_keys[v] for v in input_aa_list[4:] if v>=0]
    linker_flag = 0

    for i in range(len(AA_key_list)):
        AA_key = AA_key_list[i]
        print(AA_key)

        if AA_key == 'X2':
            #### Main N-C ####
            c_joint_mol = Chem.MolFromSmiles('[1*]C')
            reaction_pattern = '[*:1][*' + str(i+1+200) + '].[*1][*:2] >> [*:1][*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([joint_MC_mol, c_joint_mol])
            joint_MC_mol = x[0][0]
            #### Main Ca-R ####
            h_joint_mol = Chem.MolFromSmiles('[1*][H]')
            reaction_pattern = '[*:1][*' + str(i+1) + '].[*1][*:2] >> [*:1][*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
            joint_MC_mol = x[0][0]
            #### Main Ca-H ####
            h_joint_mol = Chem.MolFromSmiles('[1*][H]')
            reaction_pattern = '[*:1][*' + str(i+1+100) + '].[*1][*:2] >> [*:1][*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
            joint_MC_mol = x[0][0]

        elif AA_key == 'P':
            aa_joint = '[50*]CCC[51*]'
            aa_joint_mol = Chem.MolFromSmiles(aa_joint)
            reaction_pattern = '[*:1][*' + str(i + 1 + 200) + '].[*50][*:2] >> [*:1][*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([joint_MC_mol, aa_joint_mol])
            joint_MC_mol = x[0][0]
            reaction_pattern = '([*:1]-[' + str(i + 1) + '*].[51*]-[*:2])>>[*:1]-[*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([joint_MC_mol])
            joint_MC_mol = x[0][0]
            #### Main Ca-H ####
            h_joint_mol = Chem.MolFromSmiles('[1*][H]')
            reaction_pattern = '[*:1][*' + str(i + 1 + 100) + '].[*1][*:2] >> [*:1][*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
            joint_MC_mol = x[0][0]

        else:
            if isinstance(AA_dict[AA_key], list):
                if AA_dict[AA_key][1] == 'x':
                    continue

                elif AA_dict[AA_key][1] == 'a':
                    #### Main N-H ####
                    h_joint_mol = Chem.MolFromSmiles('[1*][H]')
                    reaction_pattern = '[*:1][*' + str(i + 1 + 200) + '].[*1][*:2] >> [*:1][*:2]'
                    rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                    x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
                    joint_MC_mol = x[0][0]
                    #### Main Ca-R ####
                    aa_joint = AA_joint[AA_key]
                    aa_joint_mol = Chem.MolFromSmiles(aa_joint)
                    reaction_pattern = '[*:1][*' + str(i + 1) + '].[*1][*:2] >> [*:1][*:2]'
                    rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                    x = rxn.RunReactants([joint_MC_mol, aa_joint_mol])
                    joint_MC_mol = x[0][0]
                    #### Main Ca-H ####
                    h_joint_mol = Chem.MolFromSmiles('[1*][H]')
                    reaction_pattern = '[*:1][*' + str(i + 1 + 100) + '].[*1][*:2] >> [*:1][*:2]'
                    rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                    x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
                    joint_MC_mol = x[0][0] 

                elif AA_dict[AA_key][1] == 'cyclic':
                    #### Main N-H ####
                    h_joint_mol = Chem.MolFromSmiles('[1*][H]')
                    reaction_pattern = '[*:1][*' + str(i + 1 + 200)+'].[*1][*:2] >> [*:1][*:2]'
                    rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                    x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
                    joint_MC_mol = x[0][0]
                    aa_joint = '[50*]' + AA_joint[AA_key] + '[51*]'
                    aa_joint_mol = Chem.MolFromSmiles(aa_joint)
                    reaction_pattern = '[*:1][*' + str(i + 1) + '].[*50][*:2] >> [*:1][*:2]'
                    rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                    x = rxn.RunReactants([joint_MC_mol, aa_joint_mol])
                    joint_MC_mol = x[0][0]
                    reaction_pattern = '([*:1]-[' + str(i + 1 + 100) + '*].[51*]-[*:2])>>[*:1]-[*:2]'
                    rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                    x = rxn.RunReactants([joint_MC_mol])
                    joint_MC_mol = x[0][0]

                elif AA_dict[AA_key][1] == 'a_a':
                     #### Main N-H ####
                    h_joint_mol = Chem.MolFromSmiles('[1*][H]')
                    reaction_pattern = '[*:1][*' + str(i + 1 + 200) + '].[*1][*:2] >> [*:1][*:2]'
                    rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                    x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
                    joint_MC_mol = x[0][0]
                    c_joint_mol_a = Chem.MolFromSmiles('[1*]' + AA_joint[AA_key][0])
                    c_joint_mol_b = Chem.MolFromSmiles('[1*]' + AA_joint[AA_key][1])
                    #### Main Ca-R ####
                    reaction_pattern = '[*:1][*' + str(i + 1) + '].[*1][*:2] >> [*:1][*:2]'
                    rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                    x = rxn.RunReactants([joint_MC_mol, c_joint_mol_a])
                    joint_MC_mol = x[0][0]
                    #### Main Ca-H ####
                    reaction_pattern = '[*:1][*' + str(i + 1 + 100) + '].[*1][*:2] >> [*:1][*:2]'
                    rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                    x = rxn.RunReactants([joint_MC_mol, c_joint_mol_b])
                    joint_MC_mol = x[0][0]
                
                elif AA_dict[AA_key][1] == 'staple':
                    #### Main N-H ####
                    h_joint_mol = Chem.MolFromSmiles('[1*][H]')
                    reaction_pattern = '[*:1][*' + str(i + 1 + 200)+'].[*1][*:2] >> [*:1][*:2]'
                    rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                    x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
                    joint_MC_mol = x[0][0]

                    if AA_key == 'S5': 
                        aa_joint = '[1*]CCCC=[300*]'
                        aa_joint_mol = Chem.MolFromSmiles(aa_joint)
                        reaction_pattern = '[*:1][*' + str(i + 1) + '].[*1][*:2] >> [*:1][*:2]'
                        rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                        x = rxn.RunReactants([joint_MC_mol, aa_joint_mol])
                        joint_MC_mol = x[0][0]

                        if linker_flag == 1:
                            reaction_pattern = '([C:1111][C:1]=[300*].[300*]=[C:2][C:2222])>>[C:1111]/[*:1]=[*:2]\[C:2222]'
                            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                            x = rxn.RunReactants([joint_MC_mol])
                            joint_MC_mol = x[0][0]
                            joint_MC_mol.UpdatePropertyCache(strict=False)

                        c_joint_mol = Chem.MolFromSmiles('[1*]C')
                        reaction_pattern = '[*:1][*' + str(i + 1 + 100) + '].[*1][*:2] >> [*:1][*:2]'
                        rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                        x = rxn.RunReactants([joint_MC_mol, c_joint_mol])
                        joint_MC_mol = x[0][0]
                        linker_flag = 1

                    elif AA_key == 'R8': 
                        aa_joint = '[1*]CCCCCCC=[300*]'
                        aa_joint_mol = Chem.MolFromSmiles(aa_joint)
                        reaction_pattern = '[*:1][*' + str(i + 1 + 100) + '].[*1][*:2] >> [*:1][*:2]'
                        rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                        x = rxn.RunReactants([joint_MC_mol, aa_joint_mol])
                        joint_MC_mol = x[0][0]

                        if linker_flag == 1:
                            reaction_pattern = '([C:1111][*:1]=[300*].[300*]=[*:2][C:2222])>>[C:1111]/[*:1]=[*:2]/[C:2222]'
                            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                            x = rxn.RunReactants([joint_MC_mol])
                            joint_MC_mol = x[0][0]
                            joint_MC_mol.UpdatePropertyCache(strict=False)

                        c_joint_mol = Chem.MolFromSmiles('[1*]C')
                        reaction_pattern = '[*:1][*' + str(i + 1) + '].[*1][*:2] >> [*:1][*:2]'
                        rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                        x = rxn.RunReactants([joint_MC_mol, c_joint_mol])
                        joint_MC_mol = x[0][0]
                        linker_flag = 1
                    
                    else:
                        print('error')

                else:
                    print('error')

            else:
                #### Main N-H ####
                h_joint_mol = Chem.MolFromSmiles('[1*][H]')
                reaction_pattern = '[*:1][*' + str(i + 1 + 200) + '].[*1][*:2] >> [*:1][*:2]'
                rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
                joint_MC_mol = x[0][0]
                #### Main Ca-R ####
                aa_joint = AA_joint[AA_key]
                aa_joint_mol = Chem.MolFromSmiles(aa_joint)
                reaction_pattern = '[*:1][*' + str(i + 1) + '].[*1][*:2] >> [*:1][*:2]'
                rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                x = rxn.RunReactants([joint_MC_mol, aa_joint_mol])
                joint_MC_mol = x[0][0]
                #### Main Ca-H ####
                h_joint_mol = Chem.MolFromSmiles('[1*][H]')
                reaction_pattern = '[*:1][*' + str(i + 1 + 100) + '].[*1][*:2] >> [*:1][*:2]'
                rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
                joint_MC_mol = x[0][0]

    joint_MC_mol = Chem.RemoveHs(joint_MC_mol)
    return Chem.MolToSmiles(joint_MC_mol), joint_MC_mol


def generate_new_peptitde(base_index, input_aa_list, peptide_feature_list, smiles_list, AA_dict, AA_joint):
    pep_len = len([v for v in  peptide_feature_list[base_index][4:] if v >= 0])
    base_smiles = smiles_list[base_index]
    base_mol = Chem.MolFromSmiles(base_smiles)
    base_mol = Chem.AddHs(base_mol)
    MC_mol = Chem.MolFromSmiles('NCC(=O)'*(pep_len))
    matches = base_mol.GetSubstructMatches(MC_mol)[0]
    joint_MC_smi, joint_MC_mol = make_joint_MC(base_mol, MC_mol, pep_len)
    AA_keys = list(AA_dict.keys())
    peptide_smi, peptide_mol = make_new_peptide(joint_MC_mol, AA_keys, AA_joint, input_aa_list, AA_dict)
    return peptide_smi, peptide_mol

