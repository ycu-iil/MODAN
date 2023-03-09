from rdkit import Chem
from rdkit.Chem import AllChem

from mol_handler import get_HbondIdx


def calc_smiles_skip_connection(smi, peptide_feature, skip = 4):
    pep_len = len([v for v in  peptide_feature[4:] if v >= 0])
    mol = Chem.MolFromSmiles(smi)
    linker_count = 0
    while mol.HasSubstructMatch(Chem.MolFromSmarts('[O][C:1]@[C:2]=[C:3]@[C:4][O]')): 
        bis = mol.GetSubstructMatches(Chem.MolFromSmarts('[O][C:1]@[C:2]=[C:3]@[C:4][O]'))
        for bs in bis:
            bs = [mol.GetBondBetweenAtoms(bs[2],bs[3]).GetIdx()]
            fragments_mol = Chem.FragmentOnBonds(mol,bs,addDummies=False)
        try:
            fragments = Chem.GetMolFrags(fragments_mol,asMols=True)
        except:
            return None
        mol = fragments[0]
        linker_count += 1

    MC_smiles = Chem.MolFromSmiles('NC(=O)C'*(pep_len))

    matches = mol.GetSubstructMatches(MC_smiles)[0]

    vertical_list = [Chem.MolFromSmiles('[H][1*]')]*skip
    for i in range(pep_len):
        skip_base = i % skip

        if i < pep_len - skip:
            bs = [mol.GetBondBetweenAtoms(matches[i*4 + 1],matches[i*4 + 3]).GetIdx(), mol.GetBondBetweenAtoms(matches[i*4 + 3],matches[i*4 + 4]).GetIdx()]
            fragments_mol = Chem.FragmentOnBonds(mol,bs,addDummies=True,dummyLabels=[(int(i/skip+1), int(i/skip+1)), (int(i/skip + 2), int(i/skip + 2))])
            fragments = Chem.GetMolFrags(fragments_mol,asMols=True)
            for fragment in fragments:
              if '['+str(int(i/skip+1))+'*]' in Chem.MolToSmiles(fragment) and '['+str(int(i/skip+2))+'*]' in Chem.MolToSmiles(fragment):
                aa_fragment = fragment
                break

            reaction_pattern = '[*:1]['+str(int(i/skip+1))+'*].['+str(int(i/skip+1))+'*][*:2] >> [*:1][*:2]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            vertical_list[skip_base] = rxn.RunReactants([vertical_list[skip_base], aa_fragment])[0][0]
        else:
            reaction_pattern = '[*:1]['+str(int(i/skip+1))+'*] >> [*:1][H]'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            vertical_list[skip_base] = rxn.RunReactants([vertical_list[skip_base]])[0][0]
    vertical_list = [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in vertical_list]
    return vertical_list


def replaceX_smiles(smi, peptide_feature, base_atom = 'P'):
    pep_len = len([v for v in  peptide_feature[4:] if v >= 0])
    mol = Chem.MolFromSmiles(smi)
    tmp = Chem.MolFromSmiles('NC(=O)C'*(pep_len))  
    mc_pattern, pc_pattern = '', ''
    for i in range(pep_len):
        mc_pattern += '[N:'+str(i*4+1)+'][C:'+str(i*4+2)+'](=[O:'+str(i*4+3)+'])[C:'+str(i*4+4)+']'
        pc_pattern += '[N:'+str(i*4+1)+'][C:'+str(i*4+2)+'](=[O:'+str(i*4+3)+'])['+base_atom+':'+str(i*4+4)+']([H])([H])'

    reaction_pattern = mc_pattern + '>>' + pc_pattern
    rxn = AllChem.ReactionFromSmarts(reaction_pattern)
    x = rxn.RunReactants([mol])[0]
    return Chem.MolToSmiles(x[0])

def calc_graph_connect(smi, peptide_feature, skip = 4):
    pep_len = len([v for v in  peptide_feature[4:] if v >= 0])
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    pc_pattern = 'NC(=O)P'*pep_len
    matches = mol.GetSubstructMatches(Chem.MolFromSmiles(pc_pattern))[0]

  
    for i in range(pep_len - skip):
        matches = mol.GetSubstructMatches(Chem.MolFromSmiles(pc_pattern))[0]
        bs = [get_HbondIdx(mol, i*4+3, matches), get_HbondIdx(mol,  i*4+ 3 + skip*4, matches)]
        fragments_mol = Chem.FragmentOnBonds(mol,bs,addDummies=True,dummyLabels=[(1,1), (2,2)])
        fragments = Chem.GetMolFrags(fragments_mol,asMols=True)
        reaction_pattern = "([*1]-[P:1].[P:2]-[*2])>>[P:1]-[P:2]"
        rxn = AllChem.ReactionFromSmarts(reaction_pattern)
        mol = rxn.RunReactants([fragments[0]])[0][0]
    x = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(x)

def calc_graph_connect_S(smi, peptide_feature, skip = 4):
    pep_len = len([v for v in  peptide_feature[4:] if v >= 0])
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    pc_pattern = 'NC(=O)S'*pep_len
    matches = mol.GetSubstructMatches(Chem.MolFromSmiles(pc_pattern))[0]
    def get_HbondIdx(mol, p_index):
        for bond in mol.GetAtomWithIdx(matches[p_index]).GetBonds():
            if bond.GetEndAtom().GetSymbol() == 'H':
                return mol.GetBondBetweenAtoms(matches[p_index],bond.GetEndAtom().GetIdx()).GetIdx()

    for i in range(pep_len - skip):
        matches = mol.GetSubstructMatches(Chem.MolFromSmiles(pc_pattern))[0]
        bs = [get_HbondIdx(mol, i*4+3), get_HbondIdx(mol,  i*4+ 3 + skip*4)]
        fragments_mol = Chem.FragmentOnBonds(mol,bs,addDummies=True,dummyLabels=[(1,1), (2,2)])
        fragments = Chem.GetMolFrags(fragments_mol,asMols=True)
        reaction_pattern = "([*1]-[S:1].[S:2]-[*2])>>[S:1]-[S:2]"
        rxn = AllChem.ReactionFromSmarts(reaction_pattern)
        mol = rxn.RunReactants([fragments[0]])[0][0]
    x = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(x)


def calc_smiles_woMC(smi, peptide_feature, base_atom = 'P'):
    pep_len = len([v for v in  peptide_feature[4:] if v >= 0])
    mol = Chem.MolFromSmiles(smi)
    tmp = Chem.MolFromSmiles('NC(=O)C'*(pep_len))
    linker_count = 0
    while mol.HasSubstructMatch(Chem.MolFromSmarts('[O][C:1]@[C:2]=[C:3]@[C:4][O]')): 

        bis = mol.GetSubstructMatches(Chem.MolFromSmarts('[O][C:1]@[C:2]=[C:3]@[C:4][O]'))
        print(bis)
        for bs in bis:
            bs = [mol.GetBondBetweenAtoms(bs[2],bs[3]).GetIdx()]
            print(bs)
            fragments_mol = Chem.FragmentOnBonds(mol,bs,addDummies=True,dummyLabels=[(300+linker_count, 300+linker_count)])
  
        try:
            fragments = Chem.GetMolFrags(fragments_mol,asMols=True)
        except:
            return None
  
        mol = fragments[0]
        linker_count += 1
  
    #Aib
    Aib_count = 0
    while mol.HasSubstructMatch(Chem.MolFromSmarts('[C][C&h0](!@[C])(N)C=O')): 
        bis = mol.GetSubstructMatches(Chem.MolFromSmarts('[C][C&h0](!@[C])(N)C=O'))
        print(bis)
        bs = bis[0]
        bond = [mol.GetBondBetweenAtoms(bs[1],bs[2]).GetIdx(), mol.GetBondBetweenAtoms(bs[0],bs[1]).GetIdx()]
        fragments_mol = Chem.FragmentOnBonds(mol,bond,addDummies=True,dummyLabels=[(400+Aib_count, 400++Aib_count), (400+Aib_count, 400+Aib_count)])
        fragments = Chem.GetMolFrags(fragments_mol,asMols=True)
        mol = fragments[0]

        Aib_count += 1

    #Ac6c
    Ac6c_count = 0
    while mol.HasSubstructMatch(Chem.MolFromSmarts('NC1(CCCCC1)C=O')): 
        bis = mol.GetSubstructMatches(Chem.MolFromSmarts('NC1(CCCCC1)C=O'))
        print(bis)
        bs = bis[0]
        print(bs)
        bond = [mol.GetBondBetweenAtoms(bs[1],bs[2]).GetIdx()]
        print(bond)
        fragments_mol = Chem.FragmentOnBonds(mol,bond,addDummies=True,dummyLabels=[(600+Ac6c_count, 600+Ac6c_count)])
  
        fragments = Chem.GetMolFrags(fragments_mol,asMols=True)
        mol = fragments[0]
        Ac6c_count += 1

  
    #Ac5c
    Ac5c_count = 0
    while mol.HasSubstructMatch(Chem.MolFromSmarts('NC1(CCCC1)C=O')): 
        bis = mol.GetSubstructMatches(Chem.MolFromSmarts('NC1(CCCC1)C=O'))
        print(bis)
        bs = bis[0]
        print(bs)
        bond = [mol.GetBondBetweenAtoms(bs[1],bs[2]).GetIdx()]
        print(bond)
        fragments_mol = Chem.FragmentOnBonds(mol,bond,addDummies=True,dummyLabels=[(500+Ac5c_count, 500+Ac5c_count)])
  
        fragments = Chem.GetMolFrags(fragments_mol,asMols=True)
        mol = fragments[0]
        Ac5c_count += 1
    print(Chem.MolToSmiles(mol))
  

    matches = mol.GetSubstructMatches(tmp)
    print(len(matches), matches)
    rep_core = AllChem.ReplaceCore(mol, tmp)

    side_mol = Chem.GetMolFrags(rep_core, asMols=True)

    print([Chem.MolToSmiles(x) for x in side_mol])
    side_smi_list = [Chem.MolToSmiles(x) for x in side_mol]
    ordered_side_smi_list = []
    for i in range(len(side_mol)):
        for side_smi in side_smi_list:
            if '['+str(i+1)+'*]' in side_smi:
                ordered_side_smi_list.append(side_smi)
                break
    print(ordered_side_smi_list)
    ordered_side_mol_list = [Chem.MolFromSmiles(smi) for smi in ordered_side_smi_list]

    for i in range(len(side_mol)):
        if i == 0:
            if base_atom == 'P':
                reaction_pattern = '[*:1][*:100].[*:2][*:101] >> [*201][P]([*:100])([*:101])'
            elif base_atom == 'C':
                reaction_pattern = '[*:1][*:100].[*:2][*:101] >> [*201][C]([*:100])([*:101])'
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([ordered_side_mol_list[0], ordered_side_mol_list[1]])[0][0]
            print(i, Chem.MolToSmiles(x), Chem.MolToSmarts(x))
        elif i < len(side_mol) - 2:
            if base_atom == 'P':
                reaction_pattern = '[*:'+str(i+2)+'][*:100].[*:'+str(200+i)+'][P:101] >>  [*'+str(200+i+1)+'][P]([*:100])([P:101])' #  [P:20'+str(i+1)+']([*:100])[P:20'+str(i)+'][*:101]'
            elif base_atom == 'C':
                reaction_pattern = '[*:'+str(i+2)+'][*:100].[*:'+str(200+i)+'][C:101] >>  [*'+str(200+i+1)+'][C]([*:100])([C:101])' #  [P:20'+str(i+1)+']([*:100])[P:20'+str(i)+'][*:101]'

            print(i, reaction_pattern)
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([ordered_side_mol_list[i+1], x])[0][0]
            print(i, Chem.MolToSmiles(x), Chem.MolToSmarts(x))
        elif i == len(side_mol) - 2:
            if base_atom == 'P':
                reaction_pattern = '[*:'+str(i+2)+'][*:100].[*:'+str(200+i)+'][P:101] >>  [P]([*:100])([P:101])'
            elif base_atom == 'C':
                reaction_pattern = '[*:'+str(i+2)+'][*:100].[*:'+str(200+i)+'][C:101] >>  [C]([*:100])([C:101])'
            print(i, reaction_pattern)
            rxn = AllChem.ReactionFromSmarts(reaction_pattern)
            x = rxn.RunReactants([ordered_side_mol_list[i+1], x])[0][0]
            print(i, Chem.MolToSmiles(x))
  
    #linker
    for count in range(linker_count):
        
        rxn = AllChem.ReactionFromSmarts('([*:1]=['+str(300+count)+'*].['+str(300+count)+'*]=[*:2])>>[*:1]=[*:2]')
        x = rxn.RunReactants([x])[0][0]
  

    #Aib
    for count in range(Aib_count):
        
        if x.HasSubstructMatch(Chem.MolFromSmarts('['+str(400+count)+'*][P:1][P:2]['+str(400+count)+'*]')):
            rxn = AllChem.ReactionFromSmarts('['+str(400+count)+'*][P:1][P:2](['+str(400+count)+'*])[P:3]>>[C][P:1]([C])[P:3]')
            x = rxn.RunReactants([x])[0][0]
        
        if x.HasSubstructMatch(Chem.MolFromSmarts('['+str(400+count)+'*][P:1]['+str(400+count)+'*]')):
            rxn = AllChem.ReactionFromSmarts('['+str(400+count)+'*][P:1]['+str(400+count)+'*]>>[C][P:1][C]')
            x = rxn.RunReactants([x])[0][0]
  
    #Ac6c
    for count in range(Ac6c_count):
        
        if x.HasSubstructMatch(Chem.MolFromSmarts('['+str(600+count)+'*][P:1]([P:3])[P:4][C:5][C:6][C:7][C:8][C:9]['+str(600+count)+'*]')):
            rxn = AllChem.ReactionFromSmarts('['+str(600+count)+'*][P:1]([P:3])[P:4][C:5][C:6][C:7][C:8][C:9]['+str(600+count)+'*]>>[P:3][P:4]1([C:5][C:6][C:7][C:8][C:9]1)')
            x = rxn.RunReactants([x])[0][0]
        
        if x.HasSubstructMatch(Chem.MolFromSmarts('['+str(600+count)+'*][P:1][C:5][C:6][C:7][C:8][C:9]['+str(600+count)+'*]')):
            rxn = AllChem.ReactionFromSmarts('['+str(600+count)+'*][P:1][C:5][C:6][C:7][C:8][C:9]['+str(600+count)+'*]>>[P:1]1([C:5][C:6][C:7][C:8][C:9]1)')
            x = rxn.RunReactants([x])[0][0]
    #Ac5c
    for count in range(Ac5c_count):

        if x.HasSubstructMatch(Chem.MolFromSmarts('['+str(500+count)+'*][P:1]([P:3])[P:4][C:5][C:6][C:7][C:8]['+str(500+count)+'*]')):
            rxn = AllChem.ReactionFromSmarts('['+str(500+count)+'*][P:1]([P:3])[P:4][C:5][C:6][C:7][C:8]['+str(500+count)+'*]>>[P:3][P:4]1([C:5][C:6][C:7][C:8]1)')
            x = rxn.RunReactants([x])[0][0]

        if x.HasSubstructMatch(Chem.MolFromSmarts('['+str(500+count)+'*][P:1][C:5][C:6][C:7][C:8]['+str(500+count)+'*]')):
            rxn = AllChem.ReactionFromSmarts('['+str(500+count)+'*][P:1][C:5][C:6][C:7][C:8]['+str(500+count)+'*]>>[P:1]1([C:5][C:6][C:7][C:8]1)')
            x = rxn.RunReactants([x])[0][0]


    return Chem.MolToSmiles(x)