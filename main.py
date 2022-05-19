#!/usr/bin/env python
# coding: utf-8

# # 準備

# In[1]:


#!pip install japanize-matplotlib
#!pip install optuna
import pandas as pd
import matplotlib.pyplot as plt
#import japanize_matplotlib 
import re
import unicodedata
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import lightgbm as lgb
import math
import optuna

import copy
import scipy.stats as stats

#RDKitの準備. とりあえず実行してください. 少々時間がかかります. 
#!pip install kora
#import kora.install.rdkit
from rdkit import rdBase
print(rdBase.rdkitVersion)

#RDKitと関連するライブラリをインポート
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw, Descriptors
#以下は特に描画用
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG


#!pip install mordred
from mordred import Calculator, descriptors


# # データ読み込み
from google.colab import drive
drive.mount('/content/drive')
data = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/data/抗菌ペプチド情報_共同研究(寺山先生)_出水_修正版20220322.xlsx')


# In[8]:


data = pd.read_excel('./data/抗菌ペプチド情報_共同研究(寺山先生)_出水_修正版20220322.xlsx')


# In[9]:


data

data['Δ[θ] ([θ]222/[θ]208)']
# # 前処理

# In[14]:


peptide_list = data['修正ペプチド配列']
for p in peptide_list:
    print(p)


# In[ ]:





# In[11]:


smiles_list = data['SMILES']
mol_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]


# アミノ酸記号http://nomenclator.la.coocan.jp/chem/text/aminosym.htm
# 
# 修正方針
# 
# - 基本はL体ということで,大文字はL体, 小文字はD体とする
# 
# - U=Aib (全体的に統一されていたのでOK)
# 
# - 一箇所(71番)Z=AibがあったのでZ->Uへ
# 
# - Z=Ac6c (全体的に統一されていたのでOK)
# 
# - X0=L-homoserine-(O-allyl) (No.11,12)
# 
# - X1 = Dab (No.45,51,52,61,62)
# 
# - X2 = Sarcosine (No.)
# 
# - 架橋している場合, アミノ酸文字の後に=を入れる
#   -- S5,S8の間には架橋が入る
# 
# - B=Ac5c (No.14)
# 
# - S5 = (S)-2-(4-pentenyl)Alanine
# 
# - R8 = (R)-2-(7-pentenyl)Alanine
# 
# - Orn (Orthinine?)が含まれている配列があるのでOに置き換え
# 
# 

# 特徴量への変換方針
# 
# 2次構造予測スコアを特徴量に入れてしまう.
# 
# 分割方法.
# - まず'-'で分割. C末端, N末端は別処理. 
# - AA配列の分
#   - 数字が入っているか確認し, もし入っていたら数字入り要素を分割
#   - 数字入り要素以外を1文字づつ分割
# 
#   - 用意したAAリストでindexを割り振る
# 
# - 架橋は別フラグを用意する.
# 
# - 特徴量の全体は, C末端情報, N末端情報, AA配列情報(indexの後に架橋しているかのbit,D体(0)L体(1)のbit, の繰り返し) 
# 
# 

# In[12]:


#とりあえずL体だけで用意
AA_dict = {
  'A': 'Alanine',
  'C': 'Cysteine',
  'D': 'Aspartic acid',
  'E': 'Glutamic acid',
  'F': 'Phenylalanine',
  'G': 'Glycine',
  'H': 'Histidine',
  'I': 'Isoleucine',
  'K': 'Lysine',
  'L': 'Leucine',
  'M': 'Methionine',
  'N': 'Asparagine',
  'P': 'Proline',
  'Q': 'Glutamine',
  'R': 'Arginine',
  'S': 'Serine',
  'T': 'Threonine',
  'V': 'Valine',
  'W': 'Tryptophane',
  'Y': 'Tyrosine',
  'O': 'Orthinine',
  'X0': 'L-homoserine-(O-allyl)',
  'X1': 'Dab',
  'X2': 'Sarcosine',
  'B': 'Ac5c',
  'U': 'Aib',
  'Z': 'Ac6c',
  'S5': '(S)-2-(4-pentenyl)Alanine',
  'R8': '(R)-2-(7-pentenyl)Alanine',
}
#D体用に拡張
D_AA_dict = {}
for aa in AA_dict.keys():
  D_AA_dict[aa.lower()] = 'D-'+AA_dict[aa]

AA_dict.update(D_AA_dict)  # MEMO: 混乱するのでAA_all_dict とかにする

#架橋用
AA_dict['='] = 'Link'


# In[13]:


AA_keys = list(AA_dict.keys())
link_index_list = []
for st in ['S5', 'R8', 's5', 'r8', '=']:
  link_index_list.append(AA_keys.index(st))
print('link_index_list', link_index_list)


SR_index_list = []
for st in ['S', 'R', 's', 'r']:
  SR_index_list.append(AA_keys.index(st))
print('SR_index_list', SR_index_list)


# In[15]:


ct_list, nt_list = [], []
for peptide in peptide_list:
  #remove '\xa0' and ' ' in peptide string
  peptide = unicodedata.normalize("NFKD", peptide).strip()
  ct,aa_list,nt = peptide.split('-')
  ct_list.append(ct)
  nt_list.append(nt)
ct_list = list(set(ct_list))
nt_list = list(set(nt_list))
print('ct_list', ct_list)
print('nt_list', nt_list)


# In[16]:


peptide_feature_list = []
#for peptide in peptide_list[35:36]:
for peptide in peptide_list:
  print(peptide)
  peptide = unicodedata.normalize("NFKD", peptide).strip()
  ct,aa_list,nt = peptide.split('-')
  ct_index = ct_list.index(ct)
  nt_index = nt_list.index(nt)
  
  print(aa_list)
  ##Indexing AA-sequence
  tmp_list = []
  for i, AA_key in enumerate(AA_dict.keys()):
    res = re.finditer(AA_key, aa_list)
    for s in res:
        tmp_list.append([s.span()[0], i])
        print(i, AA_key, s.span()[0])
  tmp_list = sorted(tmp_list, key=lambda x:float(x[0]))

  #'S', 'S5'等の重複削除
  print('tmp_list', tmp_list)
  new_tmp_list = []
  for tmp in tmp_list:
    if tmp[0]+1 < len(aa_list):
      if tmp[1] in SR_index_list:
        if aa_list[tmp[0]+1] in ['5', '8']:
          continue
    new_tmp_list.append(tmp)
  tmp_list = new_tmp_list
  print('removed_tmp_list', tmp_list)
 

  AA_index_list = []
  link_list = []

  for pair in tmp_list:
    if pair[1] in link_index_list:
      link_list.append(len(AA_index_list)+1)
    if pair[1] not in [AA_keys.index('=')]:
      AA_index_list.append(pair[1])

  if len(link_list) == 0:
    link_list = [-1, -1]
  peptide_feature = [ct_index, nt_index] + link_list + AA_index_list
  print(peptide_feature)
  peptide_feature_list.append(peptide_feature)


# In[17]:


def peptide_feature2AA_seq(pf):
  aa_seq = ''
  for j, k in enumerate(pf[4:]):
    if j in pf[2:4]:
      aa_seq += '='
    aa_seq += AA_keys[k]
  seq = ct_list[pf[0]]+'-'+aa_seq+'-'+nt_list[pf[1]]
  return seq


# In[19]:


#AA_keys = list(AA_dict.keys())

for i, pf in enumerate(peptide_feature_list):
  
  seq = peptide_feature2AA_seq(pf)
  print(i, seq)
  print(i, peptide_list[i])
  print('')

#Check

AA_keys = list(AA_dict.keys())

for i, pf in enumerate(peptide_feature_list):
  aa_seq = ''
  for j, k in enumerate(pf[4:]):
    if j in pf[2:4]:
      aa_seq += '='
    aa_seq += AA_keys[k]
  
  seq = ct_list[pf[0]]+'-'+aa_seq+'-'+nt_list[pf[1]]
  print(i, seq)
  print(i, peptide_list[i])
  print('')
# In[37]:


#padding
max_len = np.max([len(v) for v in peptide_feature_list])
print('max_len', max_len)
for peptide_feature in peptide_feature_list:
  pad_len = max_len - len(peptide_feature)
  peptide_feature += [-2] * pad_len

for fl in peptide_feature_list:
  print(fl)
# # 新規ペプチド生成
# 

# In[42]:


#アミノ酸構造の準備 #Prolineは未対応
#L-体
AA_joint = {
  'A': '[1*]C',
  'C': '[1*]CS',
  'D': 'OC(=O)C[1*]',
  'E': 'OC(=O)CC[1*]',
  'F': '[1*]CC1=CC=CC=C1',
  'G': '[1*][H]',
  'H': '[1*]CC1=CN=CN1',
  'I': 'CC[C@H](C)[1*]',
  'K': 'NCCCC[1*]',
  'L': 'CC(C)C[1*]',
  'M': 'CSCC[1*]',
  'N': 'NC(=O)C[1*]',
  'P': 'Proline', #Prolineは未対応
  'Q': 'NC(=O)CC[1*]',
  'R': 'NC(N)=NCCC[1*]',
  'S': 'OC[1*]',
  'T': 'C[C@@H](O)[1*]',
  'V': 'CC(C)[1*]',
  'W': '[1*]CC1=CNC2=CC=CC=C12',
  'Y': 'OC1=CC=C(C[1*])C=C1',
  'O': 'NCCC[1*]',
  'X0': '[1*]CCOCC=C',
  'X1': 'NCC[1*]', 
  'X2': 'Sarcosine', #要対応 Nにメチル, CaのところはのところはGlycineと同じ([*1]H)
  'B': 'C1CC[1*]C1', #要対応 5員環にする
  'U': 'Aib', # 要対応 C(C)(C)にする
  'Z': 'C1CC[1*]CC1', # 要対応6員環にする 
  'S5': 'C[1*]CCC\C=[300*]', #向きは難しい...自信ない C[*1]CCCC=[*300]
  'R8': 'C[1*]CCCCCCC=[300*]', #C[*1]CCCCCCC=[*300] 
}

#{
#    'D-I': 'CC[C@@H](C)[*1]'
#}


# In[43]:


def make_joint_MC(base_mol, MC_mol, pep_len):

  for i in range(pep_len):
    #print(i)

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
    
    
    h_idx = [x.GetIdx() for x in Ca_atom.GetNeighbors() if x.GetAtomicNum() == 1][0]
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


# In[44]:


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

#'S5': 'C[1*]CCC\C=[300*]', #向きは難しい...自信ない C[*1]CCCC=[*300]
#'R8': 'C[1*]CCCCCCC=[300*]', #C[*1]CCCCCCC=[*300] 

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


# In[45]:


def generate_new_peptitde(base_index, input_aa_list):

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


# In[47]:


#出水先生に指定してもらったデータ
#番号9, H-GIKKFLKSAKKFVKAFK-NH2, 
#番号番号69 H-KLLKKAGKLLKKAGKLLKKAG-NH2 

#baseの指定と, 生成したいaa配列を指定する
#大前提: linker (S5, R8)は配列中に合わせて0個か2個しか出現しない.
#プロリンには対応できていない.
#L体のみに対応.

base_index = 8
#B:24, U:25, Z:26, S5:27, R8:28, 
input_aa_list = peptide_feature_list[base_index]
new_peptide_smi, new_peptide_mol = generate_new_peptitde(base_index, input_aa_list)


# In[46]:


#グリシン -H  [1*]-[H]
#アラニン -CH3 [1*]-C
#Lysine -CCCCN [1*]-CCCCN


#ベースになるものとして, Glyでないくっつ
def mol_with_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol


# In[ ]:





# # 分子構造編集

# In[48]:


def calc_smiles_skip_connection(smi, peptide_feature, skip = 4):
  
  pep_len = len([v for v in  peptide_feature[4:] if v >= 0])
  #print(pep_len, peptide_feature, smi)
  mol = Chem.MolFromSmiles(smi)

  #if the peptide has a linker, then cut it.
  linker_count = 0
  while mol.HasSubstructMatch(Chem.MolFromSmarts('[O][C:1]@[C:2]=[C:3]@[C:4][O]')): 
    bis = mol.GetSubstructMatches(Chem.MolFromSmarts('[O][C:1]@[C:2]=[C:3]@[C:4][O]'))
    #print(bis)
    for bs in bis:
      bs = [mol.GetBondBetweenAtoms(bs[2],bs[3]).GetIdx()]
      #print(bs)
      fragments_mol = Chem.FragmentOnBonds(mol,bs,addDummies=False)
    try:
      fragments = Chem.GetMolFrags(fragments_mol,asMols=True)
    except:
      return None
    mol = fragments[0]
    linker_count += 1

  #print('mol', Chem.MolToSmiles(mol))
  MC_smiles = Chem.MolFromSmiles('NC(=O)C'*(pep_len))

  #print(MC_smiles)

  matches = mol.GetSubstructMatches(MC_smiles)[0]
  #print(len(matches), matches)

  vertical_list = [Chem.MolFromSmiles('[H][1*]')]*skip
  #print([Chem.MolToSmiles(mol) for mol in vertical_list])
  #print(vertical_list)
  for i in range(pep_len):
    skip_base = i % skip
    #print(i, skip_base, i/skip)

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
      #print('[*:1]['+str(int(i/skip+1))+'*].['+str(int(i/skip+1))+'*][H] >> [*:1][H]')
      reaction_pattern = '[*:1]['+str(int(i/skip+1))+'*] >> [*:1][H]'
      rxn = AllChem.ReactionFromSmarts(reaction_pattern)
      vertical_list[skip_base] = rxn.RunReactants([vertical_list[skip_base]])[0][0]
    #print([Chem.MolToSmiles(mol) for mol in vertical_list])
  vertical_list = [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in vertical_list]
  return vertical_list


# In[49]:


def calc_feature_skip_connection(smi, peptide_feature, skip, feature, descriptor_dimension = 2048):
  vertical_list = calc_smiles_skip_connection(smi, peptide_feature, skip = 4)
  
 
  vertical_feature_list = []
  for mol in vertical_list:
    #print(Chem.MolToSmiles(mol))
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
  #print(vertical_feature_list)
  vertical_feature = np.mean(vertical_feature_list, axis = 0)
  return vertical_feature


# In[50]:


def replaceP_smiles(smi, peptide_feature, base_atom = 'P'):
  pep_len = len([v for v in  peptide_feature[4:] if v >= 0])
  mol = Chem.MolFromSmiles(smi)
  tmp = Chem.MolFromSmiles('NC(=O)C'*(pep_len))
  
  #print('[N:1][C:2](=[O:3])[C:4] >> [N:1][C:2](=[O:3])[P:4]')
  mc_pattern, pc_pattern = '', ''
  for i in range(pep_len):
    mc_pattern += '[N:'+str(i*4+1)+'][C:'+str(i*4+2)+'](=[O:'+str(i*4+3)+'])[C:'+str(i*4+4)+']'
    pc_pattern += '[N:'+str(i*4+1)+'][C:'+str(i*4+2)+'](=[O:'+str(i*4+3)+'])[P:'+str(i*4+4)+']([H])([H])'

  reaction_pattern = mc_pattern + '>>' + pc_pattern
  rxn = AllChem.ReactionFromSmarts(reaction_pattern)
  x = rxn.RunReactants([mol])[0]
  #print(x)
  #print(Chem.MolToSmiles(x[0]))
  return Chem.MolToSmiles(x[0])


# In[52]:


def calc_graph_connect(smi, peptide_feature, skip = 4):
  pep_len = len([v for v in  peptide_feature[4:] if v >= 0])
  mol = Chem.MolFromSmiles(smi)
  mol = Chem.AddHs(mol)
  Chem.SanitizeMol(mol)
  #print('H', Chem.MolToSmiles(mol))

  
  pc_pattern = 'NC(=O)P'*pep_len
  #print(pc_pattern)
  matches = mol.GetSubstructMatches(Chem.MolFromSmiles(pc_pattern))[0]
  #print(len(matches), pep_len, matches)

  

  def get_HbondIdx(mol, p_index):
    for bond in mol.GetAtomWithIdx(matches[p_index]).GetBonds():
      if bond.GetEndAtom().GetSymbol() == 'H':
        return mol.GetBondBetweenAtoms(matches[p_index],bond.GetEndAtom().GetIdx()).GetIdx()

  for i in range(pep_len - skip):
  #print('matches[0],matches[1]', matches[0],matches[1])
  #print([mol.GetBondBetweenAtoms(matches[3],bond.GetEndAtom().GetIdx()).GetIdx() for bond in mol.GetAtomWithIdx(matches[3]).GetBonds() if bond.GetEndAtom().GetSymbol() == 'H'])
    matches = mol.GetSubstructMatches(Chem.MolFromSmiles(pc_pattern))[0]
    #print(len(matches), i*4+3, i*4+ 3 + skip*4)
    bs = [get_HbondIdx(mol, i*4+3), get_HbondIdx(mol,  i*4+ 3 + skip*4)]
    #bs = [mol.GetBondBetweenAtoms(matches[3],bond.GetEndAtom().GetIdx()).GetIdx() for bond in mol.GetAtomWithIdx(matches[3]).GetBonds() if bond.GetEndAtom().GetSymbol() == 'H']
    fragments_mol = Chem.FragmentOnBonds(mol,bs,addDummies=True,dummyLabels=[(1,1), (2,2)])
    fragments = Chem.GetMolFrags(fragments_mol,asMols=True)
    #print(Chem.MolToSmiles(Chem.RemoveHs(fragments[0])))
    reaction_pattern = "([*1]-[P:1].[P:2]-[*2])>>[P:1]-[P:2]" #"([C:1]=[C;H2].[C:2]=[C;H2])>>[*:1]=[*:2]" #" [1*][*:1].[*:2][2*] >> [*:1][*:2]"
    rxn = AllChem.ReactionFromSmarts(reaction_pattern)
    mol = rxn.RunReactants([fragments[0]])[0][0]
    #print(mol)
  x = Chem.RemoveHs(mol)
  #print(Chem.MolToSmiles(x))
  return Chem.MolToSmiles(x)
  """

  matches = mol.GetSubstructMatches(MC_smiles)[0]
  
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
  """


# In[ ]:





# In[53]:


def calc_smiles_woMC(smi, peptide_feature, base_atom = 'P'):
  pep_len = len([v for v in  peptide_feature[4:] if v >= 0])
  mol = Chem.MolFromSmiles(smi)
  tmp = Chem.MolFromSmiles('NC(=O)C'*(pep_len))

  #if the peptide has a linker, then cut it.
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
  
  #Aib, Ac6cについてもlinkerのような処理を行う. 一旦切っておいてラベルをつけておき最後に付け直す 
  #Aib
  Aib_count = 0
  while mol.HasSubstructMatch(Chem.MolFromSmarts('[C][C&h0](!@[C])(N)C=O')): 
    bis = mol.GetSubstructMatches(Chem.MolFromSmarts('[C][C&h0](!@[C])(N)C=O'))
    print(bis)
    bs = bis[0]
    #bond = [mol.GetBondBetweenAtoms(bs[0],bs[1]).GetIdx()]
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
  Draw.MolsToGridImage([x for x in side_mol])

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
  


  #一番最後にloop, メチル化(Aib(25), Ac6c(26)の処理)
  #linker
  for count in range(linker_count):
    #通常処理
    rxn = AllChem.ReactionFromSmarts('([*:1]=['+str(300+count)+'*].['+str(300+count)+'*]=[*:2])>>[*:1]=[*:2]')
    x = rxn.RunReactants([x])[0][0]
  

  #Aib
  #if x.HasSubstructMatch(Chem.MolFromSmarts('['+str(400+count)+'*][P:1]([P:3])[P:4][C:5][C:6][C:7][C:8][C:9]['+str(400+count)+'*]')):
  for count in range(Aib_count):
    #通常処理
    if x.HasSubstructMatch(Chem.MolFromSmarts('['+str(400+count)+'*][P:1][P:2]['+str(400+count)+'*]')):
      rxn = AllChem.ReactionFromSmarts('['+str(400+count)+'*][P:1][P:2](['+str(400+count)+'*])[P:3]>>[C][P:1]([C])[P:3]')
      x = rxn.RunReactants([x])[0][0]
    #終端にいる場合の処理
    if x.HasSubstructMatch(Chem.MolFromSmarts('['+str(400+count)+'*][P:1]['+str(400+count)+'*]')):
      rxn = AllChem.ReactionFromSmarts('['+str(400+count)+'*][P:1]['+str(400+count)+'*]>>[C][P:1][C]')
      x = rxn.RunReactants([x])[0][0]
  
  #Ac6c
  for count in range(Ac6c_count):
    #通常処理
    if x.HasSubstructMatch(Chem.MolFromSmarts('['+str(600+count)+'*][P:1]([P:3])[P:4][C:5][C:6][C:7][C:8][C:9]['+str(600+count)+'*]')):
      rxn = AllChem.ReactionFromSmarts('['+str(600+count)+'*][P:1]([P:3])[P:4][C:5][C:6][C:7][C:8][C:9]['+str(600+count)+'*]>>[P:3][P:4]1([C:5][C:6][C:7][C:8][C:9]1)')
      x = rxn.RunReactants([x])[0][0]
    #終端にいる場合の処理
    if x.HasSubstructMatch(Chem.MolFromSmarts('['+str(600+count)+'*][P:1][C:5][C:6][C:7][C:8][C:9]['+str(600+count)+'*]')):
      rxn = AllChem.ReactionFromSmarts('['+str(600+count)+'*][P:1][C:5][C:6][C:7][C:8][C:9]['+str(600+count)+'*]>>[P:1]1([C:5][C:6][C:7][C:8][C:9]1)')
      x = rxn.RunReactants([x])[0][0]
  #Ac5c
  for count in range(Ac5c_count):
    #通常処理
    if x.HasSubstructMatch(Chem.MolFromSmarts('['+str(500+count)+'*][P:1]([P:3])[P:4][C:5][C:6][C:7][C:8]['+str(500+count)+'*]')):
      rxn = AllChem.ReactionFromSmarts('['+str(500+count)+'*][P:1]([P:3])[P:4][C:5][C:6][C:7][C:8]['+str(500+count)+'*]>>[P:3][P:4]1([C:5][C:6][C:7][C:8]1)')
      x = rxn.RunReactants([x])[0][0]
    #終端にいる場合の処理
    if x.HasSubstructMatch(Chem.MolFromSmarts('['+str(500+count)+'*][P:1][C:5][C:6][C:7][C:8]['+str(500+count)+'*]')):
      rxn = AllChem.ReactionFromSmarts('['+str(500+count)+'*][P:1][C:5][C:6][C:7][C:8]['+str(500+count)+'*]>>[P:1]1([C:5][C:6][C:7][C:8]1)')
      x = rxn.RunReactants([x])[0][0]
    

  return Chem.MolToSmiles(x)
  #Draw.MolsToGridImage([ordered_side_mol_list[0], ordered_side_mol_list[1], x])
  


# In[54]:


#for i in range(len(smiles_list)):
#5(Ac6c),8(主鎖検出でエラー, 元のSMILES修正でOK), 11(Aib, 架橋),13(Ac5c)
smiles_woMC_list = []
for i in range(len(smiles_list)):
  print(i, smiles_list[i])
  seq_smi = calc_smiles_woMC(smiles_list[i], peptide_feature_list[i])
  smiles_woMC_list.append(seq_smi)


# In[55]:


smiles_repP_list = []
for i in range(len(smiles_list)):
  print(i, smiles_list[i])
  seq_smi = replaceP_smiles(smiles_list[i], peptide_feature_list[i])
  smiles_repP_list.append(seq_smi)


# # 特徴量計算

# In[56]:


#Calculation of Fingerprint, descriptor
descriptor_dimension = 1024

def calc_MorganCount(mol, r = 2, dimension = 2048):
  info = {}
  _fp = AllChem.GetMorganFingerprint(mol, r, bitInfo=info)
  count_list = [0] * dimension
  for key in info:
    pos = key % dimension
    count_list[pos] += len(info[key])
  return count_list

radial = 4

#original smiles
Morgan_r2_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, descriptor_dimension) for mol in mol_list]
Morgan_r4_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, radial, descriptor_dimension) for mol in mol_list]
MACCS_fp = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mol_list]
Morgan_r2_count = [calc_MorganCount(mol, 2, descriptor_dimension) for mol in mol_list]
Morgan_r4_count = [calc_MorganCount(mol, radial, descriptor_dimension) for mol in mol_list]

#smiles_woMC
mol_woMC_list = [Chem.MolFromSmiles(smi) for smi in smiles_woMC_list]
woMC_Morgan_r2_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, descriptor_dimension) for mol in mol_woMC_list]
woMC_Morgan_r4_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, radial, descriptor_dimension) for mol in mol_woMC_list]
woMC_MACCS_fp = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mol_woMC_list]
woMC_Morgan_r2_count = [calc_MorganCount(mol, 2, descriptor_dimension) for mol in mol_woMC_list]
woMC_Morgan_r4_count = [calc_MorganCount(mol, radial, descriptor_dimension) for mol in mol_woMC_list]

#smiles_repP
mol_repP_list = [Chem.MolFromSmiles(smi) for smi in smiles_repP_list]
repP_Morgan_r2_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, descriptor_dimension) for mol in mol_repP_list]
repP_Morgan_r4_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, radial, descriptor_dimension) for mol in mol_repP_list]
repP_MACCS_fp = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mol_repP_list]
repP_Morgan_r2_count = [calc_MorganCount(mol, 2, descriptor_dimension) for mol in mol_repP_list]
repP_Morgan_r4_count = [calc_MorganCount(mol, radial, descriptor_dimension) for mol in mol_repP_list]

#smiles_repP_skip4
mol_repP_skip4_list = [Chem.MolFromSmiles(calc_graph_connect(smi, peptide_feature, skip = 4)) for smi, peptide_feature in zip(smiles_repP_list, peptide_feature_list)]
repP_skip4_Morgan_r2_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, descriptor_dimension) for mol in mol_repP_skip4_list]
repP_skip4_Morgan_r4_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, radial, descriptor_dimension) for mol in mol_repP_skip4_list]
repP_skip4_MACCS_fp = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mol_repP_skip4_list]
repP_skip4_Morgan_r2_count = [calc_MorganCount(mol, 2, descriptor_dimension) for mol in mol_repP_skip4_list]
repP_skip4_Morgan_r4_count = [calc_MorganCount(mol, radial, descriptor_dimension) for mol in mol_repP_skip4_list]

#smiles_repP_skip7
mol_repP_skip7_list = [Chem.MolFromSmiles(calc_graph_connect(smi, peptide_feature, skip = 7)) for smi, peptide_feature in zip(smiles_repP_list, peptide_feature_list)]
repP_skip7_Morgan_r2_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, descriptor_dimension) for mol in mol_repP_skip7_list]
repP_skip7_Morgan_r4_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, radial, descriptor_dimension) for mol in mol_repP_skip7_list]
repP_skip7_MACCS_fp = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mol_repP_skip7_list]
repP_skip7_Morgan_r2_count = [calc_MorganCount(mol, 2, descriptor_dimension) for mol in mol_repP_skip7_list]
repP_skip7_Morgan_r4_count = [calc_MorganCount(mol, radial, descriptor_dimension) for mol in mol_repP_skip7_list]




# In[ ]:


#vertical_feature
"""
v_skip4_Morgan_r2_fp = [calc_feature_skip_connection(smiles_list[i], peptide_feature_list[i], skip = 4, feature = 'Morgan_r2', descriptor_dimension = descriptor_dimension) for i in range(len(smiles_list))]
v_skip4_Morgan_r4_fp = [calc_feature_skip_connection(smiles_list[i], peptide_feature_list[i], skip = 4, feature = 'Morgan_r4', descriptor_dimension = descriptor_dimension) for i in range(len(smiles_list))]
v_skip4_Morgan_r2_count = [calc_feature_skip_connection(smiles_list[i], peptide_feature_list[i], skip = 4, feature = 'Morgan_r2_count', descriptor_dimension = descriptor_dimension) for i in range(len(smiles_list))]
v_skip4_Morgan_r4_count = [calc_feature_skip_connection(smiles_list[i], peptide_feature_list[i], skip = 4, feature = 'Morgan_r4_count', descriptor_dimension = descriptor_dimension) for i in range(len(smiles_list))]
v_skip4_MACCS_fp = [calc_feature_skip_connection(smiles_list[i], peptide_feature_list[i], skip = 4, feature = 'MACCS') for i in range(len(smiles_list))]

v_skip7_Morgan_r2_fp = [calc_feature_skip_connection(smiles_list[i], peptide_feature_list[i], skip = 7, feature = 'Morgan_r2', descriptor_dimension = descriptor_dimension) for i in range(len(smiles_list))]
v_skip7_Morgan_r4_fp = [calc_feature_skip_connection(smiles_list[i], peptide_feature_list[i], skip = 7, feature = 'Morgan_r4', descriptor_dimension = descriptor_dimension) for i in range(len(smiles_list))]
v_skip7_Morgan_r2_count = [calc_feature_skip_connection(smiles_list[i], peptide_feature_list[i], skip = 7, feature = 'Morgan_r2_count', descriptor_dimension = descriptor_dimension) for i in range(len(smiles_list))]
v_skip7_Morgan_r4_count = [calc_feature_skip_connection(smiles_list[i], peptide_feature_list[i], skip = 7, feature = 'Morgan_r4_count', descriptor_dimension = descriptor_dimension) for i in range(len(smiles_list))]
v_skip7_MACCS_fp = [calc_feature_skip_connection(smiles_list[i], peptide_feature_list[i], skip = 7, feature = 'MACCS') for i in range(len(smiles_list))]
"""


# In[57]:


def calc_mordred_descriptor(mol_list):
  #mordred
  calc = Calculator(descriptors, ignore_3D = True)
  mordred_df = calc.pandas(mol_list)

  #modredののerrorをNaNで置き換え
  df_descriptors = mordred_df.astype(str)
  masks = df_descriptors.apply(lambda d: d.str.contains('[a-zA-Z]' ,na=False))
  df_descriptors = df_descriptors[~masks]
  df_descriptors = df_descriptors.astype(float)
  modred_descriptor = df_descriptors.dropna(axis=1, how='any')

  return modred_descriptor


# In[ ]:


#mordred_descriptor = calc_mordred_descriptor(mol_list)
#woMC_mordred_descriptor = calc_mordred_descriptor(mol_woMC_list)
#repP_mordred_descriptor = calc_mordred_descriptor(mol_repP_list)
#repP_skip4_mordred_descriptor = calc_mordred_descriptor(mol_repP_skip4_list)
#repP_skip7_mordred_descriptor = calc_mordred_descriptor(mol_repP_skip7_list)

#v_skip4_mordred_descriptor = [calc_feature_skip_connection(smiles_list[i], peptide_feature_list[i], skip = 4, feature = 'mordred') for i in range(len(smiles_list))]
#v_skip7_mordred_descriptor = [calc_feature_skip_connection(smiles_list[i], peptide_feature_list[i], skip = 7, feature = 'mordred') for i in range(len(smiles_list))]


# # 予測モデル構築準備

# In[58]:


#physboのインポート
#pipでインストール
#!pip3 install physbo
import physbo

from sklearn import preprocessing 


# In[ ]:





# In[59]:


def GP_predict(train_X, test_X, train_y, test_y):

  cov = physbo.gp.cov.gauss(train_X,ard = False )
  mean = physbo.gp.mean.const()
  lik = physbo.gp.lik.gauss()
  gp = physbo.gp.model(lik=lik,mean=mean,cov=cov)
  config = physbo.misc.set_config()

  gp.fit(train_X, train_y, config)
  gp.print_params()
  gp.prepare(train_X, train_y)

  train_fmean = gp.get_post_fmean(train_X, train_X) 
  train_fcov = gp.get_post_fcov(train_X, train_X)

  test_fmean = gp.get_post_fmean(train_X, test_X) 
  test_fcov = gp.get_post_fcov(train_X, test_X)

  return [train_fmean, train_fcov], [test_fmean, test_fcov] 


# In[64]:


def calc_prediction_model(smiles_type, model, feature, fold_n, target_index, value_log = False, standardize = False):

  target_name = data.keys()[target_index]
  exp_list = data[target_name][:82]
  print(target_name)

  #数値データの修正
  filled_index_list = []
  exp_modified_list = []
  for i, v in enumerate(exp_list):
    if str(v)[0] == '>':
      exp_modified_list.append(float(str(v)[1:])*2)
      filled_index_list.append(i)
    elif str(v)[0] == '<':
      exp_modified_list.append(float(str(v)[1:])/2)
      filled_index_list.append(i)
    else:
      if not math.isnan(v):
        filled_index_list.append(i)
      exp_modified_list.append(v)
  print(len(filled_index_list))
  if value_log == True:
    exp_modified_list = np.log10(exp_modified_list)
  plt.hist(np.array(exp_modified_list)[filled_index_list])
  plt.title(target_name+' Log10='+str(value_log))
  plt.xlabel(target_name)
  plt.ylabel('frequency')
  if target_name == 'Δ[θ] ([θ]222/[θ]208)':
    plt.savefig('./result/helix-like_dist_log'+str(value_log)+'.png', dpi = 300)
  else:
    plt.savefig('./result/'+target_name+'_dist_log'+str(value_log)+'.png', dpi = 300)
  plt.show()


  print('feature', feature, smiles_type)
  if smiles_type == 'original':
    if feature == 'one-hot':
      X = np.array(peptide_feature_list)[filled_index_list]
    elif feature == 'Morgan_r2':
      X = np.array(Morgan_r2_fp)[filled_index_list]
    elif feature == 'Morgan_r4':
      X = np.array(Morgan_r4_fp)[filled_index_list]
    elif feature == 'MACCS':
      X = np.array(MACCS_fp)[filled_index_list]
    elif feature == 'mordred':
      X = mordred_descriptor.values[filled_index_list]
    elif feature == 'Morgan_r2_count':
      X = np.array(Morgan_r2_count)[filled_index_list]
    elif feature == 'Morgan_r4_count':
      X = np.array(Morgan_r4_count)[filled_index_list]
    elif feature == 'MACCS+Morgan_r4_count':
      X0 = np.array(MACCS_fp)[filled_index_list]
      X1 = np.array(Morgan_r4_count)[filled_index_list]
      X = np.concatenate([X0, X1], axis = 1)
  if smiles_type == 'smiles_woMC':
    if feature == 'Morgan_r2':
      X = np.array(woMC_Morgan_r2_fp)[filled_index_list]
    elif feature == 'Morgan_r4':
      X = np.array(woMC_Morgan_r4_fp)[filled_index_list]
    elif feature == 'MACCS':
      X = np.array(woMC_MACCS_fp)[filled_index_list]
    elif feature == 'mordred':
      X = woMC_mordred_descriptor.values[filled_index_list]
    elif feature == 'Morgan_r2_count':
      X = np.array(woMC_Morgan_r2_count)[filled_index_list]
    elif feature == 'Morgan_r4_count':
      X = np.array(woMC_Morgan_r4_count)[filled_index_list]
  if smiles_type == 'smiles_repP':
    if feature == 'Morgan_r2':
      X = np.array(repP_Morgan_r2_fp)[filled_index_list]
    elif feature == 'Morgan_r4':
      X = np.array(repP_Morgan_r4_fp)[filled_index_list]
    elif feature == 'MACCS':
      X = np.array(repP_MACCS_fp)[filled_index_list]
    elif feature == 'mordred':
      X = repP_mordred_descriptor.values[filled_index_list]
    elif feature == 'Morgan_r2_count':
      X = np.array(repP_Morgan_r2_count)[filled_index_list]
    elif feature == 'Morgan_r4_count':
      X = np.array(repP_Morgan_r4_count)[filled_index_list]
  if smiles_type == 'smiles_repP_skip4':
    if feature == 'Morgan_r2':
      X = np.array(repP_skip4_Morgan_r2_fp)[filled_index_list]
    elif feature == 'Morgan_r4':
      X = np.array(repP_skip4_Morgan_r4_fp)[filled_index_list]
    elif feature == 'MACCS':
      X = np.array(repP_skip4_MACCS_fp)[filled_index_list]
    elif feature == 'mordred':
      X = repP_skip4_mordred_descriptor.values[filled_index_list]
    elif feature == 'Morgan_r2_count':
      X = np.array(repP_skip4_Morgan_r2_count)[filled_index_list]
    elif feature == 'Morgan_r4_count':
      X = np.array(repP_skip4_Morgan_r4_count)[filled_index_list]
  if smiles_type == 'smiles_repP_skip7':
    if feature == 'Morgan_r2':
      X = np.array(repP_skip7_Morgan_r2_fp)[filled_index_list]
    elif feature == 'Morgan_r4':
      X = np.array(repP_skip7_Morgan_r4_fp)[filled_index_list]
    elif feature == 'MACCS':
      X = np.array(repP_skip7_MACCS_fp)[filled_index_list]
    elif feature == 'mordred':
      X = repP_skip7_mordred_descriptor.values[filled_index_list]
    elif feature == 'Morgan_r2_count':
      X = np.array(repP_skip7_Morgan_r2_count)[filled_index_list]
    elif feature == 'Morgan_r4_count':
      X = np.array(repP_skip7_Morgan_r4_count)[filled_index_list]
  if smiles_type == 'vertical_skip7':
    if feature == 'Morgan_r2':
      X = np.array(v_skip7_Morgan_r2_fp)[filled_index_list]
    elif feature == 'Morgan_r4':
      X = np.array(v_skip7_Morgan_r4_fp)[filled_index_list]
    elif feature == 'MACCS':
      X = np.array(v_skip7_MACCS_fp)[filled_index_list]
    elif feature == 'mordred':
      X = np.array(v_skip7_mordred_descriptor)[filled_index_list]
    elif feature == 'Morgan_r2_count':
      X = np.array(v_skip7_Morgan_r2_count)[filled_index_list]
    elif feature == 'Morgan_r4_count':
      X = np.array(v_skip7_Morgan_r4_count)[filled_index_list]
  if smiles_type == 'vertical_skip4':
    if feature == 'Morgan_r2':
      X = np.array(v_skip4_Morgan_r2_fp)[filled_index_list]
    elif feature == 'Morgan_r4':
      X = np.array(v_skip4_Morgan_r4_fp)[filled_index_list]
    elif feature == 'MACCS':
      X = np.array(v_skip4_MACCS_fp)[filled_index_list]
    elif feature == 'mordred':
      X = np.array(v_skip4_mordred_descriptor)[filled_index_list]
    elif feature == 'Morgan_r2_count':
      X = np.array(v_skip4_Morgan_r2_count)[filled_index_list]
    elif feature == 'Morgan_r4_count':
      X = np.array(v_skip4_Morgan_r4_count)[filled_index_list]


  """
  PCA_dim_reduction = True
  print(100, len(X[0]))
  #n_components = np.min([100, len(X[0])])
  n_components = 35
  if PCA_dim_reduction:
    # PCAで次元削減
    print('PCA calc')
    pca = PCA(n_components=n_components, svd_solver='arpack')
    X = pca.fit_transform(X)
  """

  if model == 'physbo' and standardize:
    
    ss = preprocessing.StandardScaler()
    X = ss.fit_transform(X)
  
  y = np.array(exp_modified_list)[filled_index_list]

  print(X)
  print(y)

  from sklearn.model_selection import KFold
  kf = KFold(n_splits = fold_n, shuffle = True, random_state=0)

  y_pred_list = []
  y_test_list = []
  y_index_list = []
    
  def objective(trial):
    params = {
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100)
    }
    train_model.set_params(**params)
    scores = cross_val_score(train_model, X_train, y_train, cv=cv,
                            scoring='neg_mean_squared_error', fit_params=fit_params, n_jobs=-1)
    return scores.mean()
    
  for train_index, test_index in kf.split(X):
    print(train_index, test_index)
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    if model == 'RF':
      train_model = RandomForestRegressor()
      train_model.fit(X_train, y_train)
    elif model == 'lightgbm':
      seed = 0
      train_model = lgb.LGBMRegressor(boosting_type='gbdt', objective='regression',
                      random_state=seed, n_estimators=100)
      fit_params = {
          'verbose': 2,
          'early_stopping_rounds': 10,
          'eval_metric': 'mean_squared_error',
          'eval_set': [(X_train, y_train)]
          }
      cv = KFold(n_splits=5, shuffle=True, random_state=seed)
      study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=seed))
      study.optimize(objective, n_trials=200)
      best_params = study.best_trial.params
      train_model.set_params(**best_params)
      train_model.fit(X_train, y_train)


      #train_model = lgb.LGBMRegressor() # モデルのインスタンスの作成
      #train_model.fit(X_train, y_train) # モデルの学習
      y_pred = train_model.predict(X_test)
      y_train_pred = train_model.predict(X_train)

      y_pred_list += list(y_pred)
      y_test_list += list(y_test)

    elif model == 'physbo':
      [y_train_pred, y_train_pred_cov], [y_pred, y_pred_cov] = GP_predict(X_train, X_test, y_train, y_test)
      train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
      train_r = np.corrcoef(y_train, y_train_pred)[0][1]
      print('train_rmse', train_rmse, 'train_r', train_r)

      y_pred_list += list(y_pred)
      y_test_list += list(y_test)


    y_index_list += list(test_index)
    #r2 = r2_score(y_test, y_pred),
    #rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    #print(r2, rmse)
    r = np.corrcoef(y_test_list, y_pred_list)[0][1]
    r2 = r2_score(y_test_list, y_pred_list)
    #auc = roc_auc_score(y_test_list, y_pred_list)
    rmse = np.sqrt(mean_squared_error(y_test_list, y_pred_list))
    print(len(y_pred_list), 'r', r, 'r2', r2, 'RMSE', rmse, smiles_type, model, feature, fold_n, target_index)
    


  # Feature Importance
  #fti = clf_rf.feature_importances_   

  r = np.corrcoef(y_test_list, y_pred_list)[0][1]
  r2 = r2_score(y_test_list, y_pred_list)
  #auc = roc_auc_score(y_test_list, y_pred_list)

  rmse = np.sqrt(mean_squared_error(y_test_list, y_pred_list))
  print('RESULT', model, feature, fold_n, target_index)
  print('r2', r2, 'r', r, 'rmse', rmse)
  print('\n\n\n\n\n\n\n\n\n\n\n')

  plt.scatter(y_test_list, y_pred_list)
  plt.plot([np.min(y_test_list), np.max(y_test_list)], [np.min(y_test_list), np.max(y_test_list)])
  for i in range(len(y_test_list)):
    plt.annotate(str(i+1), (y_test_list[i]+0.02, y_pred_list[i]+0.02))
  plt.title(target_name+' (N='+str(len(y_test_list))+', model = '+model+') r='+str(round(r, 3)))
  plt.grid()
  if value_log:
    plt.xlabel('Experimental value (log)')
    plt.ylabel('Predicted value (log)')
  else:
    plt.xlabel('Experimental value')
    plt.ylabel('Predicted value')
  plt.axes().set_aspect('equal')
  if target_name == 'Δ[θ] ([θ]222/[θ]208)':
    plt.savefig('./result/helix-like_feature'+feature+'_CV'+str(fold_n)+'_model'+model+'_smile'+smiles_type+'_scatter.png', dpi = 300)
  else:
    plt.savefig('./result/'+target_name+'_feature'+feature+'_CV'+str(fold_n)+'_model'+model+'_smile'+smiles_type+'_scatter.png', dpi = 300)
  plt.show()


# # 予測精度検証

# In[65]:


#model list: 'RF', 'lightgbm'
#feature list: 'Morgan_r2', 'Morgan_r4','Morgan_r2_count', 'Morgan_r4_count', 'MACCS', 'Morgan_r2_MACCS', 'one-hot', 'mordred'

#fold_n: fold num of cross-validation

#target_index
#5:'大腸菌 (NZRC 3972)', 6:'DH5a', 7:'緑膿菌', '黄色ブドウ球菌', 'プロテウス菌', 
#'表皮ブドウ球菌', 'Proteus vulgaris', 'Salmonella enterica subsp.', 'Klebsiella pneumoniae（肺炎桿菌）', 'MDRP', 15: '溶血性', 16: Δ[θ] ([θ]222/[θ]208)

#活性値にlog10を入れるか否か否か
value_log = False
#smiles_type = 'vertical_skip4' #'original', 'smiles_repP_skip7', 'smiles_woMC', 'vertical_skip4', 'vertical_skip7'
model = 'physbo'
fold_n = 10
for smiles_type in ['smiles_repP_skip7']:
  for target_index in [16]:
    for feature in ['Morgan_r2', 'Morgan_r4', 'Morgan_r2_count', 'Morgan_r4_count', 'MACCS']:# ['Morgan_r2', 'Morgan_r4', 'Morgan_r2_count', 'Morgan_r4_count', 'MACCS', 'mordred']:#['one-hot', 'mordred', 'Morgan_r2', 'Morgan_r4', 'MACCS']:
      calc_prediction_model(smiles_type, model, feature, fold_n, target_index, value_log, standardize = False)


# In[66]:


target_index = 16
value_log = False
target_name = data.keys()[target_index]
exp_list = data[target_name][:82]
print(target_name)

#数値データの修正
filled_index_list = []
exp_modified_list = []
for i, v in enumerate(exp_list):
  if str(v)[0] == '>':
    exp_modified_list.append([float(str(v)[1:])*2, peptide_feature_list[i][2]])
    filled_index_list.append(i)
  elif str(v)[0] == '<':
    exp_modified_list.append([float(str(v)[1:])/2, peptide_feature_list[i][2]])
    filled_index_list.append(i)
  else:
    if not math.isnan(v):
      filled_index_list.append(i)
    exp_modified_list.append([v, peptide_feature_list[i][2]])

print(len(filled_index_list))
if value_log == True:
  exp_modified_list = np.log10(exp_modified_list)
max = np.max(np.array(exp_modified_list)[filled_index_list][:,0])
min = np.min(np.array(exp_modified_list)[filled_index_list][:,0])
#plt.hist(np.array(exp_modified_list)[filled_index_list][:,0], label = 'all', alpha = 0.6, bins = np.arange(min, max, (max-min)/20))
plt.hist([v[0] for v in np.array(exp_modified_list)[filled_index_list] if v[1] < 0], label = 'wo. linker', alpha = 0.6, bins = np.arange(min, max+0.01, (max-min)/20))
plt.hist([v[0] for v in np.array(exp_modified_list)[filled_index_list] if v[1] > 0], label = 'w. linker', alpha = 0.6, bins = np.arange(min, max+0.01, (max-min)/20))

#plt.hist(exp_modified_nonlinker_list, label = 'wo. linker', alpha = 0.6, bins = np.arange(0, 1.2, 0.05))
#plt.hist(exp_modified_linker_list, label = 'w. linker', alpha = 0.6, bins = np.arange(0, 1.2, 0.05))
#plt.hist(exp_modified_nonlinker_list, label = 'wo. linker', alpha = 0.6)
#plt.hist(exp_modified_linker_list, label = 'w. linker', alpha = 0.6)


plt.legend()
plt.title(target_name+' Log10='+str(value_log))
plt.xlabel(target_name)
plt.ylabel('frequency')
if target_name == 'Δ[θ] ([θ]222/[θ]208)':
  plt.savefig('./result/helix-like_dist_log'+str(value_log)+'.png', dpi = 300)
else:
  plt.savefig('./result/'+target_name+'_dist_log'+str(value_log)+'.png', dpi = 300)
plt.show()


# In[67]:


smiles_list[base_index]


# # BOによる推薦

# In[ ]:


#候補ペプチドの準備

#出水先生に指定してもらったデータ
#番号9, H-GIKKFLKSAKKFVKAFK-NH2, 
#番号番号69 H-KLLKKAGKLLKKAGKLLKKAG-NH2 

#baseの指定と, 生成したいaa配列を指定する
#実験からの要請: S5,R8に関してに関して
#1.S5~S5: 間に3つaa, 2重結合周りはcis,
#2.R8~S5: 間に6つaa, 2重結合周りはtrans
#S5R8, R8R8はなし.  
#S5ははR-にアルキルアルキル, R8はR-にメチル
#プロリン対応済み
#L体のみに対応.

#作り方の方針: linker以外をN箇所mutation, 最後に最後にlinkerをつける.


# In[68]:


import itertools

base_index = 8
input_aa_list = copy.deepcopy(peptide_feature_list[base_index])


mutation_num = 1
pep_len = len([v for v in input_aa_list[4:] if v >= 0])

NAA_index_list = list(range(21))
NNAA_index_list = [24, 25, 26] #[21, 22, 23, 24, 25, 26]
mutatable_AA_index_list = NNAA_index_list #ここどうするか
linker_index_list = [27, 28]

#linkerは入っていないと仮定. 一番最後に入れる. 最初に変異入れる箇所の候補の組み合わせを出す.
position_index_list = range(pep_len)
pos_comb_list = itertools.combinations(position_index_list, mutation_num)

#mutation_infoの構造の構造
#[linker_info, mutation_pos_list, mutation_aa_list]
mutation_info_list = [[[-1, -1], [], []]] #何も変異しないものも用意,　後にlinkerがつくことはあり 


for pos_comb in pos_comb_list:
  #print(pos_comb)
  for mutation_pos in pos_comb:
    for mutation_aa in itertools.product(mutatable_AA_index_list, repeat = mutation_num):
      mutation_info_list.append([[-1, -1], list(pos_comb), list(mutation_aa)])
      #print(pos_comb, mutation_aa)
      
print(len(mutation_info_list))

#linker をつけたものの情報を生成
linker_mutation_info_list = []
for m_i, mutation_info in enumerate(mutation_info_list):
  
  print('linker', m_i, len(mutation_info_list))
  for i in range(pep_len):
    for un in linker_index_list:
      if un == 27: #S5-S5
        if i+4 > pep_len - 1:
          continue
        new_mutation_info = copy.deepcopy(mutation_info)
        new_mutation_info[0] = [i+1, i+4+1]
        new_mutation_info[1] = new_mutation_info[1]+[i, i+4]
        new_mutation_info[2] = new_mutation_info[2]+[27, 27]
        linker_mutation_info_list.append(new_mutation_info)
        
      elif un == 28: #R8-S5
        if i+7 > pep_len - 1:
          continue
        new_mutation_info = copy.deepcopy(mutation_info)
        new_mutation_info[0] = [i+1, i+7+1]
        new_mutation_info[1] = new_mutation_info[1]+[i, i+7]
        new_mutation_info[2] = new_mutation_info[2]+[28, 27]
        linker_mutation_info_list.append(new_mutation_info)

mutation_info_list = mutation_info_list + linker_mutation_info_list
print(len(mutation_info_list))


# In[69]:


mutation_info_list[:4]


# In[70]:


new_peptide_mol_list, new_peptide_smi_list = [], []
new_peptide_feature_list = []
cand_data_list = []


for mutation_info in mutation_info_list:
  print(len(cand_data_list), len(mutation_info_list))
  input_aa_list = copy.deepcopy(peptide_feature_list[base_index])
  input_aa_list[2:4] = mutation_info[0]
  for m_pos, m_aa in zip(mutation_info[1], mutation_info[2]):
    input_aa_list[4+m_pos] = m_aa

  new_peptide_smi, new_peptide_mol = generate_new_peptitde(base_index, input_aa_list)
  cand_data_list.append([mutation_info, new_peptide_smi])
  new_peptide_feature_list.append(input_aa_list)
  new_peptide_mol_list.append(new_peptide_mol)
  new_peptide_smi_list.append(new_peptide_smi)


# In[ ]:





# In[ ]:





# In[71]:


new_smiles_repP_list = []
for i in range(len(new_peptide_smi_list)):
  print(i, new_peptide_smi_list[i])
  seq_smi = replaceP_smiles(new_peptide_smi_list[i], new_peptide_feature_list[i])
  new_smiles_repP_list.append(seq_smi)


# In[72]:


mol_list = new_peptide_mol_list

#original smiles
Cand_Morgan_r2_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, descriptor_dimension) for mol in mol_list]
Cand_Morgan_r4_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, radial, descriptor_dimension) for mol in mol_list]
Cand_MACCS_fp = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mol_list]
Cand_Morgan_r2_count = [calc_MorganCount(mol, 2, descriptor_dimension) for mol in mol_list]
Cand_Morgan_r4_count = [calc_MorganCount(mol, radial, descriptor_dimension) for mol in mol_list]

#smiles_repP_skip7
mol_repP_skip7_list = [Chem.MolFromSmiles(calc_graph_connect(smi, peptide_feature, skip = 7)) for smi, peptide_feature in zip(new_smiles_repP_list, new_peptide_feature_list)]
Cand_repP_skip7_Morgan_r2_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, descriptor_dimension) for mol in mol_repP_skip7_list]
Cand_repP_skip7_Morgan_r4_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, radial, descriptor_dimension) for mol in mol_repP_skip7_list]
Cand_repP_skip7_MACCS_fp = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mol_repP_skip7_list]
Cand_repP_skip7_Morgan_r2_count = [calc_MorganCount(mol, 2, descriptor_dimension) for mol in mol_repP_skip7_list]
Cand_repP_skip7_Morgan_r4_count = [calc_MorganCount(mol, radial, descriptor_dimension) for mol in mol_repP_skip7_list]


# In[73]:


def get_EI_list(train_Y, pred_y, sigma2_pred):
  prediction = pred_y
  sig = sigma2_pred**0.5

  gamma = (prediction - np.max(train_Y)) / sig
  ei = sig*(gamma*stats.norm.cdf(gamma) + stats.norm.pdf(gamma))

  return ei

def calc_EI_overfmax(fmean, fcov, fmax): #fmaxに基準値を入れる, fmaxに対する改善値の期待値を測る指標であることに注意.
  fstd = np.sqrt(fcov)

  temp1 = fmean - fmax
  temp2 = temp1 / fstd
  score = temp1 * stats.norm.cdf(temp2) + fstd * stats.norm.pdf(temp2)
  return score

def calc_PI_overfmax(fmean, fcov, fmax): #fmaxに基準値を入れる, fmaxを上回る確率であることに注意.
  fstd = np.sqrt(fcov)

  temp = (fmean - fmax) / fstd
  score = stats.norm.cdf(temp)
  return score

def calc_PI_underfmin(fmean, fcov, fmin): #fminに基準値を入れる, fminを下回る確率であることに注意.
  fstd = np.sqrt(fcov)

  temp = (fmin - fmean) / fstd
  score = stats.norm.cdf(temp)
  return score


# In[74]:


cand_data_list[0]


# In[76]:


#target_index
#5:'大腸菌 (NZRC 3972)', 6:'DH5a', 7:'緑膿菌', '黄色ブドウ球菌', 'プロテウス菌', 
#'表皮ブドウ球菌', 'Proteus vulgaris', 'Salmonella enterica subsp.', 'Klebsiella pneumoniae（肺炎桿菌）', 'MDRP', 15: '溶血性', 16: Δ[θ] ([θ]222/[θ]208)

target_list = [5, 6, 7, 8, 10, 14, 15]
threshold_list = [['<=', 10], ['<=', 10], ['<=', 10], ['<=', 10], ['<=', 10],  ['<=', 10], ['>=', 50]]
smiles_type_list = ['smiles_repP_skip7', 'smiles_repP_skip7', 'original', 'original', 'original', 'smiles_repP_skip7', 'smiles_repP_skip7']
model = 'physbo'
feature_list = ['MACCS', 'Morgan_r4_count', 'Morgan_r2_count', 'Morgan_r4_count', 'MACCS', 'Morgan_r4_count', 'Morgan_r4_count']
value_log = True
standardize = False
visualize = True

pred_y_list_list = []
pred_cov_list_list = []
pi_list_list = []

for target_i in range(len(target_list)):

  target_index = target_list[target_i]
  target_name = data.keys()[target_index]
  smiles_type = smiles_type_list[target_i]
  feature = feature_list[target_i]
  exp_list = data[target_name][:82]
  print(target_name)

  #数値データの修正
  filled_index_list = []
  exp_modified_list = []
  for i, v in enumerate(exp_list):
    if str(v)[0] == '>':
      exp_modified_list.append(float(str(v)[1:])*2)
      filled_index_list.append(i)
    elif str(v)[0] == '<':
      exp_modified_list.append(float(str(v)[1:])/2)
      filled_index_list.append(i)
    else:
      if not math.isnan(v):
        filled_index_list.append(i)
      exp_modified_list.append(v)
  print('filled_index_list', len(filled_index_list))
  if value_log == True:
    exp_modified_list = np.log10(exp_modified_list)

  #学習モデル準備
  #学習ペプチド入力データ準備
  if smiles_type == 'original':
    if feature == 'MACCS':
      X = np.array(MACCS_fp)[filled_index_list]
      X_cand = np.array(Cand_MACCS_fp)
    elif feature == 'Morgan_r2_count':
      X = np.array(Morgan_r2_count)[filled_index_list]
      X_cand = np.array(Cand_Morgan_r2_count)
    elif feature == 'Morgan_r4_count':
      X = np.array(Morgan_r4_count)[filled_index_list]
      X_cand = np.array(Cand_Morgan_r4_count)
  if smiles_type == 'smiles_repP_skip7':
    if feature == 'MACCS':
      X = np.array(repP_skip7_MACCS_fp)[filled_index_list]
      X_cand = np.array(Cand_repP_skip7_MACCS_fp)
    elif feature == 'Morgan_r2_count':
      X = np.array(repP_skip7_Morgan_r2_count)[filled_index_list]
      X_cand = np.array(Cand_repP_skip7_Morgan_r2_count)
    elif feature == 'Morgan_r4_count':
      X = np.array(repP_skip7_Morgan_r4_count)[filled_index_list]
      X_cand = np.array(Cand_repP_skip7_Morgan_r4_count)
      

  #候補ペプチド入力データ準備
  

  if model == 'physbo' and standardize:  
    ss = preprocessing.StandardScaler()
    X = ss.fit_transform(X)
    X_cand = ss.fit_transform(np.array(Cand_Morgan_r4_count))
  y = np.array(exp_modified_list)[filled_index_list]

  #学習
  print('X', len(X), 'X_cand', len(X_cand), 'y', len(y))
  [y_pred_train, y_pred_cov_train], [y_pred, y_pred_cov] = GP_predict(X, X_cand, y, [0 for i in range(len(X_cand))])

  #calc PI
  if threshold_list[target_i][0] == '<=':
    pi_list = calc_PI_underfmin(y_pred, y_pred_cov, np.log10(threshold_list[target_i][1]))
  elif threshold_list[target_i][0] == '>=':    
    pi_list = calc_PI_overfmax(y_pred, y_pred_cov, np.log10(threshold_list[target_i][1]))

  pred_y_list_list.append(y_pred)
  pred_cov_list_list.append(y_pred_cov)
  pi_list_list.append(pi_list)

  print('cand_data_list', cand_data_list[:5])
  print('PI', pi_list[:5])
  print('y_pred', y_pred[:5])
  print('y_cov', y_pred_cov[:5])

  if visualize:
    B_cand_index = [i for i, data in enumerate(cand_data_list) if ((len(data[0][1]) == 1) and (AA_keys[data[0][2][0]] == 'B'))]
    plt.errorbar(np.array(range(1, len(y_pred[B_cand_index]) + 1))-0.07, y_pred[B_cand_index], yerr = (y_pred_cov**0.5)[B_cand_index], fmt='o', label = 'Ac5c')

    U_cand_index = [i for i, data in enumerate(cand_data_list) if len(data[0][1]) == 1 and AA_keys[data[0][2][0]] == 'U']
    plt.errorbar(range(1, len(y_pred[U_cand_index]) + 1), y_pred[U_cand_index], yerr = (y_pred_cov**0.5)[U_cand_index], fmt='o', label = 'Aib')

    Z_cand_index = [i for i, data in enumerate(cand_data_list) if len(data[0][1]) == 1 and AA_keys[data[0][2][0]] == 'Z']
    plt.errorbar(np.array(range(1, len(y_pred[Z_cand_index]) + 1))+0.07, y_pred[Z_cand_index], yerr = (y_pred_cov**0.5)[Z_cand_index], fmt='o', label = 'Ac6c')

    S5_cand_index = [i for i, data in enumerate(cand_data_list) if len(data[0][1]) == 2 and AA_keys[data[0][2][0]] == 'S5' and AA_keys[data[0][2][1]] == 'S5']
    plt.errorbar(np.array(range(1, len(y_pred[S5_cand_index]) + 1))-0.07, y_pred[S5_cand_index], yerr = (y_pred_cov**0.5)[S5_cand_index], fmt='o', label = 'S5-S5')

    R8_cand_index = [i for i, data in enumerate(cand_data_list) if len(data[0][1]) == 2 and AA_keys[data[0][2][0]] == 'R8']
    plt.errorbar(np.array(range(1, len(y_pred[R8_cand_index]) + 1))-0.07, y_pred[R8_cand_index], yerr = (y_pred_cov**0.5)[R8_cand_index], fmt='o', label = 'R8-S5')


    plt.legend()
    plt.xlabel('mutation position')
    plt.ylabel(target_name+' (log, predicted)')
    plt.savefig('./result/bo_'+target_name+'_test.png', dpi = 300)
    plt.show()


# In[77]:


total_pi_score_list = []
for j in range(len(new_peptide_feature_list)):
  score = 1
  for i in range(len(target_list)):
    score = score*pi_list_list[i][j]
  total_pi_score_list.append(score)


# In[78]:


plt.plot(total_pi_score_list)
plt.ylabel('Total PI score')
plt.grid()
plt.show()

ordered_total_PI_score_index = np.argsort(total_pi_score_list)[::-1]

for top_index in ordered_total_PI_score_index[:10]:
  print('index', top_index, 'total_pi_score', round(total_pi_score_list[top_index],3), 'mutation_info', cand_data_list[top_index][0], peptide_feature2AA_seq([v for v in new_peptide_feature_list[top_index] if v != -2]))
  for target_i in range(len(target_list)):
    target_index = target_list[target_i]
    target_name = data.keys()[target_index]
    print('  ', target_name, round(10**pred_y_list_list[target_i][top_index], 3))
  


# In[ ]:




