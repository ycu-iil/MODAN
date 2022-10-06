from rdkit import Chem
from rdkit.Chem import AllChem


def peptide_feature2AA_seq(pf, AA_keys, ct_list, nt_list):
    aa_seq = ''
    for j, k in enumerate(pf[4:]):
        if j in pf[2:4]:
            aa_seq += '='
        aa_seq += AA_keys[k]
    seq = ct_list[pf[0]]+'-'+aa_seq+'-'+nt_list[pf[1]]
    return seq


def make_joint_MC(base_mol, MC_mol, pep_len):
    for i in range(pep_len):

        ####Ca-Cbを切って切ってjointを付ける#####
        matches = base_mol.GetSubstructMatches(MC_mol)[0]
        #print(matches)
        atom_index = matches[i*4 + 1]
        #print('atom_index', atom_index)
        Ca_atom = base_mol.GetAtomWithIdx(atom_index)
        #print(Ca_atom.GetSymbol(), atom_index)
        #print('with index', Chem.MolToSmiles(mol_with_atom_index(base_mol)))

        #for atom in Ca_atom.GetNeighbors():
        #  print('x:', atom.GetIdx(), atom.GetAtomicNum())

        c_beta_idx = [x.GetIdx() for x in Ca_atom.GetNeighbors() if x.GetIdx() not in list(matches)][0]
        atom_pair = [Ca_atom.GetIdx(), c_beta_idx]
        #print('atom_pair', atom_pair)
        bs = [base_mol.GetBondBetweenAtoms(atom_pair[0],atom_pair[1]).GetIdx()]
        #print(bs)
        fragments_mol = Chem.FragmentOnBonds(base_mol,bs,addDummies=True,dummyLabels=[(i+1,i+1)])
        try:
            fragments = Chem.GetMolFrags(fragments_mol,asMols=True)
        except:
            print("error")

        base_mol = fragments[0]

        ####Ca-Hを切って切ってjointを付ける#####
        matches = base_mol.GetSubstructMatches(MC_mol)[0]
        atom_index = matches[i*4 + 1]
        Ca_atom = base_mol.GetAtomWithIdx(atom_index)
    
    
        #h_idx = [x.GetIdx() for x in Ca_atom.GetNeighbors() if x.GetAtomicNum() == 1][0]
        h_counter = 0
        cb_counter = 0
        for x in Ca_atom.GetNeighbors() :
            if x.GetAtomicNum() == 1:
                h_counter +=1
                h_idx = x.GetIdx()
            elif (x.GetIdx() not in list(matches)) and cb_counter == 0 and x.GetAtomicNum() != 1 and h_counter == 0:
                cb_counter += 1
                h_idx = x.GetIdx()
            else:
                continue

        atom_pair = [Ca_atom.GetIdx(), h_idx]
        #print('atom_pair', atom_pair)
        bs = [base_mol.GetBondBetweenAtoms(atom_pair[0],atom_pair[1]).GetIdx()]
        #print(bs)
        fragments_mol = Chem.FragmentOnBonds(base_mol,bs,addDummies=True,dummyLabels=[(i+1+100,i+1+100)])
        try:
            fragments = Chem.GetMolFrags(fragments_mol,asMols=True)
        except:
            print("error")
        base_mol = fragments[0]

        ####N-Hを切って切ってjointを付ける#####
        matches = base_mol.GetSubstructMatches(MC_mol)[0]
        atom_index = matches[i*4 + 0]
        N_atom = base_mol.GetAtomWithIdx(atom_index)
    
    
        h_idx = [x.GetIdx() for x in N_atom.GetNeighbors() if x.GetAtomicNum() == 1][0]
        atom_pair = [N_atom.GetIdx(), h_idx]
        #print('atom_pair', atom_pair)
        bs = [base_mol.GetBondBetweenAtoms(atom_pair[0],atom_pair[1]).GetIdx()]
        #print(bs)
        fragments_mol = Chem.FragmentOnBonds(base_mol,bs,addDummies=True,dummyLabels=[(i+1+200,i+1+200)])
        try:
            fragments = Chem.GetMolFrags(fragments_mol,asMols=True)
        except:
            print("error")
        base_mol = fragments[0]

    #Chem.SanitizeMol(base_mol)
    base_mol = Chem.RemoveHs(base_mol)
    base_smi = Chem.MolToSmiles(base_mol)
    return base_smi, base_mol

def make_new_peptide(joint_MC_mol, AA_keys, AA_joint, input_aa_list):
    #各アミノ酸をつけるCaごとにX*が振られているので、アミノ酸をくっつける
    AA_key_list = [AA_keys[v] for v in input_aa_list[4:] if v>=0]


    linker_flag = 0
    for i in range(len(AA_key_list)):
        AA_key = AA_key_list[i]
        #print('i', i, 'AA_key', AA_key)

        #N-methileに絡むのはX2ととPのみなので先に処理処理
        if AA_key == 'X2':
            #### Main N-C ####
            c_joint_mol = Chem.MolFromSmiles('[1*]C')
            reaction_pattern = '[*:1][*'+str(i+1+200)+'].[*1][*:2] >> [*:1][*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([joint_MC_mol, c_joint_mol])
            joint_MC_mol = x[0][0]

            #### Main Ca-R ####
            h_joint_mol = Chem.MolFromSmiles('[1*][H]')
            reaction_pattern = '[*:1][*'+str(i+1)+'].[*1][*:2] >> [*:1][*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
            #print(x)
            joint_MC_mol = x[0][0]

            #### Main Ca-H ####
            h_joint_mol = Chem.MolFromSmiles('[1*][H]')
            reaction_pattern = '[*:1][*'+str(i+1+100)+'].[*1][*:2] >> [*:1][*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
            joint_MC_mol = x[0][0]

        elif AA_key == 'P':
            aa_joint = '[50*]CCC[51*]'
            aa_joint_mol = Chem.MolFromSmiles(aa_joint)

            reaction_pattern = '[*:1][*'+str(i+1+200)+'].[*50][*:2] >> [*:1][*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([joint_MC_mol, aa_joint_mol])
            joint_MC_mol = x[0][0]

            reaction_pattern = '([*:1]-['+str(i+1)+'*].[51*]-[*:2])>>[*:1]-[*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([joint_MC_mol])
            joint_MC_mol = x[0][0]


            #### Main Ca-H ####
            h_joint_mol = Chem.MolFromSmiles('[1*][H]')
            reaction_pattern = '[*:1][*'+str(i+1+100)+'].[*1][*:2] >> [*:1][*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
            joint_MC_mol = x[0][0]

        else:
            #### Main N-H この処理はX2以外は共通 ####
            h_joint_mol = Chem.MolFromSmiles('[1*][H]')
            reaction_pattern = '[*:1][*'+str(i+1+200)+'].[*1][*:2] >> [*:1][*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
            joint_MC_mol = x[0][0]



            #if AA_key in ['P', 'X2', 'B', 'U', 'Z', 'S5', 'R8']:
            #  print('例外処理')
            if AA_key == 'B': #Caで5員環
                aa_joint = '[50*]CCCC[51*]'
                aa_joint_mol = Chem.MolFromSmiles(aa_joint)

                reaction_pattern = '[*:1][*'+str(i+1)+'].[*50][*:2] >> [*:1][*:2]'
                rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                x = rxn.RunReactants([joint_MC_mol, aa_joint_mol])
                joint_MC_mol = x[0][0]

                reaction_pattern = '([*:1]-['+str(i+1+100)+'*].[51*]-[*:2])>>[*:1]-[*:2]'
                rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                x = rxn.RunReactants([joint_MC_mol])
                joint_MC_mol = x[0][0]

            elif AA_key == 'U': #'Aib', # 要対応 C(C)(C)にする
                c_joint_mol = Chem.MolFromSmiles('[1*]C')

                #### Main Ca-R ####

                reaction_pattern = '[*:1][*'+str(i+1)+'].[*1][*:2] >> [*:1][*:2]'
                rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                x = rxn.RunReactants([joint_MC_mol, c_joint_mol])
                joint_MC_mol = x[0][0]

                #### Main Ca-H ####
                reaction_pattern = '[*:1][*'+str(i+1+100)+'].[*1][*:2] >> [*:1][*:2]'
                rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                x = rxn.RunReactants([joint_MC_mol, c_joint_mol])
                joint_MC_mol = x[0][0]

            elif AA_key == 'Z': #Caで6員環
                aa_joint = '[50*]CCCCC[51*]'
                aa_joint_mol = Chem.MolFromSmiles(aa_joint)

                reaction_pattern = '[*:1][*'+str(i+1)+'].[*50][*:2] >> [*:1][*:2]'
                rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                x = rxn.RunReactants([joint_MC_mol, aa_joint_mol])
                joint_MC_mol = x[0][0]

                reaction_pattern = '([*:1]-['+str(i+1+100)+'*].[51*]-[*:2])>>[*:1]-[*:2]'
                rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                x = rxn.RunReactants([joint_MC_mol])
                joint_MC_mol = x[0][0]

#'  S5  ':   'C[1*]CCC\C=[300*]', #向きは難しい...自信ない C[*1]CCCC=[*300]
#'  R8  ':   'C[1*]CCCCCCC=[300*]', #C[*1]CCCCCCC=[*300] 

            elif AA_key == 'S5': 
                aa_joint = '[1*]CCCC=[300*]'
                aa_joint_mol = Chem.MolFromSmiles(aa_joint)

                reaction_pattern = '[*:1][*'+str(i+1)+'].[*1][*:2] >> [*:1][*:2]'
                rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                x = rxn.RunReactants([joint_MC_mol, aa_joint_mol])
                joint_MC_mol = x[0][0]

                if linker_flag == 1:
                  reaction_pattern = '([C:1111][C:1]=[300*].[300*]=[C:2][C:2222])>>[C:1111]/[*:1]=[*:2]\[C:2222]'
                  rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                  x = rxn.RunReactants([joint_MC_mol])
                  joint_MC_mol = x[0][0]
                  joint_MC_mol.UpdatePropertyCache(strict=False)
                  #print('check', Chem.MolToSmiles(joint_MC_mol))

                c_joint_mol = Chem.MolFromSmiles('[1*]C')
                reaction_pattern = '[*:1][*'+str(i+1+100)+'].[*1][*:2] >> [*:1][*:2]'
                rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                x = rxn.RunReactants([joint_MC_mol, c_joint_mol])
                joint_MC_mol = x[0][0]

                linker_flag = 1

            elif AA_key == 'R8': 
                aa_joint = '[1*]CCCCCCC=[300*]'
                aa_joint_mol = Chem.MolFromSmiles(aa_joint)

                reaction_pattern = '[*:1][*'+str(i+1+100)+'].[*1][*:2] >> [*:1][*:2]'
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
                reaction_pattern = '[*:1][*'+str(i+1)+'].[*1][*:2] >> [*:1][*:2]'
                rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                x = rxn.RunReactants([joint_MC_mol, c_joint_mol])
                joint_MC_mol = x[0][0]

                linker_flag = 1


            else:
                #### Main Ca-R ####
                aa_joint = AA_joint[AA_key]
                #print(aa_joint)
                aa_joint_mol = Chem.MolFromSmiles(aa_joint)

                reaction_pattern = '[*:1][*'+str(i+1)+'].[*1][*:2] >> [*:1][*:2]'
                #print(reaction_pattern)
                #print(aa_joint_mol)
                rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                x = rxn.RunReactants([joint_MC_mol, aa_joint_mol])
                #print(x)
                joint_MC_mol = x[0][0]

                #### Main Ca-H ####
                h_joint_mol = Chem.MolFromSmiles('[1*][H]')
                reaction_pattern = '[*:1][*'+str(i+1+100)+'].[*1][*:2] >> [*:1][*:2]'
                rxn = AllChem.ReactionFromSmarts(reaction_pattern)
                x = rxn.RunReactants([joint_MC_mol, h_joint_mol])
                joint_MC_mol = x[0][0]





        #print(Chem.MolToSmiles(joint_MC_mol))
    joint_MC_mol = Chem.RemoveHs(joint_MC_mol)
    #print(Chem.MolToSmiles(joint_MC_mol))

    return Chem.MolToSmiles(joint_MC_mol), joint_MC_mol


def generate_new_peptitde(base_index, input_aa_list, peptide_feature_list, smiles_list, AA_dict, AA_joint):

    pep_len = len([v for v in  peptide_feature_list[base_index][4:] if v >= 0])
    #print('pep_len', pep_len)
    base_smiles = smiles_list[base_index]

    base_mol = Chem.MolFromSmiles(base_smiles)
    base_mol = Chem.AddHs(base_mol)

    MC_mol = Chem.MolFromSmiles('NCC(=O)'*(pep_len))
    matches = base_mol.GetSubstructMatches(MC_mol)[0]
    #print(len(matches), matches)

    joint_MC_smi, joint_MC_mol = make_joint_MC(base_mol, MC_mol, pep_len)

    #print('joint_MC_smi', joint_MC_smi)

    AA_keys = list(AA_dict.keys())
    peptide_smi, peptide_mol = make_new_peptide(joint_MC_mol, AA_keys, AA_joint, input_aa_list)

    #print('peptide_smi', peptide_smi)

    return peptide_smi, peptide_mol

