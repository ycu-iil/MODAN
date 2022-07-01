import copy
import itertools
import math
import unicodedata
import re
import time
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import physbo
import pickle
import multiprocessing

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn import preprocessing 
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from acquisition_function import calc_PI_overfmax, calc_PI_underfmin
from feature_generator import calc_MorganCount, calc_mordred_descriptor
import metadata
from peptide_handler import peptide_feature2AA_seq, generate_new_peptitde
from smiles_handler import calc_smiles_skip_connection, replaceP_smiles, calc_smiles_woMC, calc_graph_connect

def generate_peptide_from_mutation_info(input_aa_list, mutation_info):
  #input_aa_list = args[0]
  #mutaion_info = args[1]
  print(input_aa_list, mutation_info)
  #リンカー情報
  input_aa_list[2:4] = mutation_info[0]
  b  = copy.copy(input_aa_list)
  c = []
  for m_pos, m_aa in zip(mutation_info[1], mutation_info[2]):
  #   #変異の挿入
  #   input_aa_list[4+m_pos] = m_aa
    input_aa_list[4+m_pos] = m_aa
    c = copy.copy(input_aa_list)
  if b == c:
    return None
    #continue  
  else:
    new_peptide_smi, new_peptide_mol = generate_new_peptitde(base_index, input_aa_list, peptide_feature_list, smiles_list, AA_dict, AA_joint)
    #new_peptide_feature_list1.append(input_aa_list)
    #cand_data_list1.append([mutation_info, new_peptide_smi])
    #new_peptide_mol_list1.append(new_peptide_mol)
    #new_peptide_smi_list1.append(new_peptide_smi)
    return new_peptide_smi, new_peptide_mol, input_aa_list, [mutation_info, new_peptide_smi]

def mol2FP(mol, fp_type, radial = 4, descriptor_dimension = 1024):
  if fp_type == 'Morgan':
    return AllChem.GetMorganFingerprintAsBitVect(mol, radial, descriptor_dimension)
  elif fp_type == 'MorganCount':
    return calc_MorganCount(mol, radial, descriptor_dimension)
  elif fp_type == 'MACCS':
    return AllChem.GetMACCSKeysFingerprint(mol)

def smi2repP_skip(smi, peptide_feature, skip = 7):
  return Chem.MolFromSmiles(calc_graph_connect(smi, peptide_feature, skip))




data = pd.read_excel('./data/AMPdata.xlsx')

#data = pd.read_excel('./data/test.xlsx')
peptide_list = data['修正ペプチド配列'][:82]
for p in peptide_list:
    print(p)
smiles_list = data['SMILES'][:82]
mol_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]


AA_dict = metadata.AA_dict
AA_joint = metadata.AA_joint

AA_keys = list(AA_dict.keys())
link_index_list = []
for st in ['S5', 'R8', 's5', 'r8', '=']:
    link_index_list.append(AA_keys.index(st))
print('link_index_list', link_index_list)


SR_index_list = []
for st in ['S', 'R', 's', 'r']:
    SR_index_list.append(AA_keys.index(st))
print('SR_index_list', SR_index_list)


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



for i, pf in enumerate(peptide_feature_list):
  
    seq = peptide_feature2AA_seq(pf, AA_keys, ct_list, nt_list)
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
new_peptide_smi, new_peptide_mol = generate_new_peptitde(base_index, input_aa_list, peptide_feature_list, smiles_list, AA_dict, AA_joint)



#グリシン -H  [1*]-[H]
#アラニン -CH3 [1*]-C
#Lysine -CCCCN [1*]-CCCCN


#for i in range(len(smiles_list)):
#5(Ac6c),8(主鎖検出でエラー, 元のSMILES修正でOK), 11(Aib, 架橋),13(Ac5c)
smiles_woMC_list = []
for i in range(len(smiles_list)):
    print(i, smiles_list[i])
    seq_smi = calc_smiles_woMC(smiles_list[i], peptide_feature_list[i])
    smiles_woMC_list.append(seq_smi)


smiles_repP_list = []
for i in range(len(smiles_list)):
    print(i, smiles_list[i])
    seq_smi = replaceP_smiles(smiles_list[i], peptide_feature_list[i])
    smiles_repP_list.append(seq_smi)


# # 特徴量計算

#Calculation of Fingerprint, descriptor
descriptor_dimension = 1024

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


# # 予測モデル構築準備

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
        elif feature == 'Morgan_r2_count':
            X = np.array(repP_skip7_Morgan_r2_count)[filled_index_list]
        elif feature == 'Morgan_r4_count':
            X = np.array(repP_skip7_Morgan_r4_count)[filled_index_list]
    """
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

    if model == 'physbo' and standardize: 
        ss = preprocessing.StandardScaler()
        X = ss.fit_transform(X)
  
    y = np.array(exp_modified_list)[filled_index_list]

    print(X)
    print(y)

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

"""
# # 予測精度検証

#model list: 'RF', 'lightgbm'
#feature list: 'Morgan_r2', 'Morgan_r4','Morgan_r2_count', 'Morgan_r4_count', 'MACCS', 'Morgan_r2_MACCS', 'one-hot'

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
        for feature in ['Morgan_r2', 'Morgan_r4', 'Morgan_r2_count', 'Morgan_r4_count', 'MACCS']:
            calc_prediction_model(smiles_type, model, feature, fold_n, target_index, value_log, standardize = False)


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
"""

# # BOによる推薦

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



base_index = 8
input_aa_list = copy.deepcopy(peptide_feature_list[base_index])

#max60くらい
proc_n = 60
mutation_num = 1 #2
pep_len = len([v for v in input_aa_list[4:] if v >= 0])

NAA_index_list = list(range(21))
NNAA_index_list = [9,17,20,22,24,25,26] #[21, 22, 23, 24, 25, 26]
mutatable_AA_index_list = NNAA_index_list #ここどうするか
linker_index_list = [27, 28]
result_type = "NNAA"  #staple, NNAA

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
linker_start_time = time.time()
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
linker_end_time = time.time()

new_peptide_mol_list1, new_peptide_smi_list1 = [], []
new_peptide_feature_list1 = []
cand_data_list1 = []


generate_start_time = time.time()

args_list = [(copy.deepcopy(peptide_feature_list[base_index]), mutation_info) for mutation_info in mutation_info_list]
print('args_list', args_list)
with multiprocessing.Pool(processes = proc_n) as pool:
  new_peptide_data_list = pool.starmap(generate_peptide_from_mutation_info, args_list)
new_peptide_smi_list1 = [data[0] for data in new_peptide_data_list if data != None]
new_peptide_mol_list1 = [data[1] for data in new_peptide_data_list if data != None]
new_peptide_feature_list1 = [data[2] for data in new_peptide_data_list if data != None]
cand_data_list1 = [data[3] for data in new_peptide_data_list if data != None]

"""
for mutation_info in mutation_info_list:
  print(len(new_peptide_feature_list1), len(mutation_info_list))
  input_aa_list = copy.deepcopy(peptide_feature_list[base_index])
  #リンカー情報
  input_aa_list[2:4] = mutation_info[0]
  b  = copy.copy(input_aa_list)
  c = []
  for m_pos, m_aa in zip(mutation_info[1], mutation_info[2]):
  #   #変異の挿入
  #   input_aa_list[4+m_pos] = m_aa
    input_aa_list[4+m_pos] = m_aa
    c = copy.copy(input_aa_list)
  if b == c:
    continue  
  else:
    new_peptide_smi, new_peptide_mol = generate_new_peptitde(base_index, input_aa_list, peptide_feature_list, smiles_list, AA_dict, AA_joint)
    new_peptide_feature_list1.append(input_aa_list)
    cand_data_list1.append([mutation_info, new_peptide_smi])
    new_peptide_mol_list1.append(new_peptide_mol)
    new_peptide_smi_list1.append(new_peptide_smi)
"""
generate_end_time = time.time()

new_peptide_mol_list, new_peptide_smi_list = [], []
new_peptide_feature_list = []
cand_data_list = []

for i, pep in enumerate(new_peptide_smi_list1):
    if pep not in new_peptide_smi_list:
        new_peptide_smi_list.append(pep)
        new_peptide_mol_list.append(new_peptide_mol_list1[i])
        new_peptide_feature_list.append(new_peptide_feature_list1[i])
        cand_data_list.append(cand_data_list1[i])

new_smiles_repP_list = []
for i in range(len(new_peptide_smi_list)):
    print(i, new_peptide_smi_list[i])
    seq_smi = replaceP_smiles(new_peptide_smi_list[i], new_peptide_feature_list[i])
    new_smiles_repP_list.append(seq_smi)


mol_list = new_peptide_mol_list

fp_start_time = time.time()
with multiprocessing.Pool(processes = proc_n) as pool:
  Cand_Morgan_r2_fp = pool.starmap(mol2FP, [(mol, 'Morgan', 2, descriptor_dimension) for mol in mol_list])

with multiprocessing.Pool(processes = proc_n) as pool:
  Cand_Morgan_r4_fp = pool.starmap(mol2FP, [(mol, 'Morgan', 4, descriptor_dimension) for mol in mol_list])

with multiprocessing.Pool(processes = proc_n) as pool:
  Cand_MACCS_fp = pool.starmap(mol2FP, [(mol, 'MACCS') for mol in mol_list])

with multiprocessing.Pool(processes = proc_n) as pool:
  Cand_Morgan_r2_count = pool.starmap(mol2FP, [(mol, 'MorganCount', 2, descriptor_dimension) for mol in mol_list])

with multiprocessing.Pool(processes = proc_n) as pool:
  Cand_Morgan_r4_count = pool.starmap(mol2FP, [(mol, 'MorganCount', 4, descriptor_dimension) for mol in mol_list])

"""
Cand_Morgan_r2_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, descriptor_dimension) for mol in mol_list]
Cand_Morgan_r4_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, radial, descriptor_dimension) for mol in mol_list]
Cand_MACCS_fp = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mol_list]
Cand_Morgan_r2_count = [calc_MorganCount(mol, 2, descriptor_dimension) for mol in mol_list]
Cand_Morgan_r4_count = [calc_MorganCount(mol, radial, descriptor_dimension) for mol in mol_list]
"""
fp_end_time = time.time()


#smiles_repP_skip7
repP_start_time = time.time()
with multiprocessing.Pool(processes = proc_n) as pool:
  mol_repP_skip7_list = pool.starmap(smi2repP_skip, [(smi, peptide_feature, 7) for smi, peptide_feature in zip(new_smiles_repP_list, new_peptide_feature_list)])

with multiprocessing.Pool(processes = proc_n) as pool:
  Cand_repP_skip7_Morgan_r2_fp = pool.starmap(mol2FP, [(mol, 'Morgan', 2, descriptor_dimension) for mol in mol_repP_skip7_list])

with multiprocessing.Pool(processes = proc_n) as pool:
  Cand_repP_skip7_Morgan_r4_fp = pool.starmap(mol2FP, [(mol, 'Morgan', 4, descriptor_dimension) for mol in mol_repP_skip7_list])

with multiprocessing.Pool(processes = proc_n) as pool:
  Cand_repP_skip7_MACCS_fp = pool.starmap(mol2FP, [(mol, 'MACCS') for mol in mol_repP_skip7_list])

with multiprocessing.Pool(processes = proc_n) as pool:
  Cand_repP_skip7_Morgan_r2_count = pool.starmap(mol2FP, [(mol, 'MorganCount', 2, descriptor_dimension) for mol in mol_repP_skip7_list])

with multiprocessing.Pool(processes = proc_n) as pool:
  Cand_repP_skip7_Morgan_r4_count = pool.starmap(mol2FP, [(mol, 'MorganCount', 4, descriptor_dimension) for mol in mol_repP_skip7_list])


"""
mol_repP_skip7_list = [Chem.MolFromSmiles(calc_graph_connect(smi, peptide_feature, skip = 7)) for smi, peptide_feature in zip(new_smiles_repP_list, new_peptide_feature_list)]
Cand_repP_skip7_Morgan_r2_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, descriptor_dimension) for mol in mol_repP_skip7_list]
Cand_repP_skip7_Morgan_r4_fp = [AllChem.GetMorganFingerprintAsBitVect(mol, radial, descriptor_dimension) for mol in mol_repP_skip7_list]
Cand_repP_skip7_MACCS_fp = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mol_repP_skip7_list]
Cand_repP_skip7_Morgan_r2_count = [calc_MorganCount(mol, 2, descriptor_dimension) for mol in mol_repP_skip7_list]
Cand_repP_skip7_Morgan_r4_count = [calc_MorganCount(mol, radial, descriptor_dimension) for mol in mol_repP_skip7_list]
"""
repP_end_time = time.time()

print('linker info time:', linker_end_time - linker_start_time)
print('smiles generation time:', generate_end_time - generate_start_time)
print('fp calc time:', fp_end_time - fp_start_time)
print('repP & fp time:', repP_end_time - repP_start_time)

#target_index
#5:'大腸菌 (NZRC 3972)', 6:'DH5a', 7:'緑膿菌', '黄色ブドウ球菌', 'プロテウス菌', 
#'表皮ブドウ球菌', 'Proteus vulgaris', 'Salmonella enterica subsp.', 'Klebsiella pneumoniae（肺炎桿菌）', 'MDRP', 15: '溶血性', 16: Δ[θ] ([θ]222/[θ]208)

target_list = [5, 6, 7, 8, 10, 14, 15]
threshold_list = [['<=', 5], ['<=', 5], ['<=', 5], ['<=', 5], ['<=', 5],  ['<=', 5], ['>=', 100]]
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


total_pi_score_list = []
for j in range(len(new_peptide_feature_list)):
    score = 1
    for i in range(len(target_list)):
        score = score*pi_list_list[i][j]
    total_pi_score_list.append(score)


plt.plot(total_pi_score_list)
plt.ylabel('Total PI score')
plt.grid()
plt.savefig('result/bo_PI score.png', dpi = 300)
plt.show()

with open('result/total_pi_score_list.pkl', mode='wb') as f:
    pickle.dump(total_pi_score_list, f)
with open('result/cand_data_list.pkl', mode='wb') as f:
    pickle.dump(cand_data_list, f)
with open('result/new_peptide_feature_list.pkl', mode='wb') as f:
    pickle.dump(new_peptide_feature_list, f)

if result_type == "staple":
    only_staple_total_pi_score_list = []
    index_list = []
    index = -1

    for i in cand_data_list:
        index += 1
        if i[0][0][0] != -1:
            index_list.append(index)
            only_staple_total_pi_score_list.append(total_pi_score_list[index])

    ordered_total_PI_score_index = np.argsort(only_staple_total_pi_score_list)[::-1]

    for top_index in ordered_total_PI_score_index[:10]:
        print( 'total_pi_score', round(only_staple_total_pi_score_list[top_index],3), 'mutation_info', cand_data_list[index_list[top_index]][0], peptide_feature2AA_seq([v for v in new_peptide_feature_list[index_list[top_index]] if v != -2],AA_keys, ct_list, nt_list))
        for target_i in range(len(target_list)):
            target_index = target_list[target_i]
            target_name = data.keys()[target_index]
            print('  ', target_name, round(10**pred_y_list_list[target_i][index_list[top_index]], 3),  '(', round(pred_cov_list_list[target_i][top_index]**0.5,3), ')' )  

elif result_type == "NNAA":
    orn_total_pi_score_list, orn_index_list = [], []
    dab_total_pi_score_list, dab_index_list = [], []
    ac5c_total_pi_score_list,ac5c_index_list = [], []
    aib_total_pi_score_list,aib_index_list = [], []
    ac6c_total_pi_score_list,ac6c_index_list = [], []
    for i, pep in enumerate(new_peptide_feature_list):
        if 20 in pep:
            orn_total_pi_score_list.append(total_pi_score_list[i])
            orn_index_list.append(i)
        
        if 22 in pep:
            dab_total_pi_score_list.append(total_pi_score_list[i])
            dab_index_list.append(i)

        if 24 in pep:
            ac5c_total_pi_score_list.append(total_pi_score_list[i])
            ac5c_index_list.append(i)

        if 25 in pep:
            aib_total_pi_score_list.append(total_pi_score_list[i])
            aib_index_list.append(i)

        if 26 in pep:
            ac6c_total_pi_score_list.append(total_pi_score_list[i])
            ac6c_index_list.append(i)
    
    
    orn_ordered_total_PI_score_index = np.argsort(orn_total_pi_score_list)[::-1]
    for top_index in orn_ordered_total_PI_score_index[:3]:
        print( 'orn','total_pi_score', round(orn_total_pi_score_list[top_index],3), 'mutation_info', cand_data_list[orn_index_list[top_index]][0], peptide_feature2AA_seq([v for v in new_peptide_feature_list[orn_index_list[top_index]] if v != -2], AA_keys, ct_list, nt_list))
        for target_i in range(len(target_list)):
            target_index = target_list[target_i]
            target_name = data.keys()[target_index]
            print('  ', target_name, round(10**pred_y_list_list[target_i][orn_index_list[top_index]], 3), '(', round(pred_cov_list_list[target_i][orn_index_list[top_index]]**0.5,3), ')' )

    dab_ordered_total_PI_score_index = np.argsort(dab_total_pi_score_list)[::-1]
    for top_index in dab_ordered_total_PI_score_index[:3]:
        print( 'dab','total_pi_score', round(dab_total_pi_score_list[top_index],3), 'mutation_info', cand_data_list[dab_index_list[top_index]][0], peptide_feature2AA_seq([v for v in new_peptide_feature_list[dab_index_list[top_index]] if v != -2], AA_keys, ct_list, nt_list))
        for target_i in range(len(target_list)):
            target_index = target_list[target_i]
            target_name = data.keys()[target_index]
            print('  ', target_name, round(10**pred_y_list_list[target_i][dab_index_list[top_index]], 3), '(', round(pred_cov_list_list[target_i][dab_index_list[top_index]]**0.5,3), ')' )
    
    ac5c_ordered_total_PI_score_index = np.argsort(ac5c_total_pi_score_list)[::-1]
    for top_index in ac5c_ordered_total_PI_score_index[:3]:
        print( 'ac5c','total_pi_score', round(ac5c_total_pi_score_list[top_index],3), 'mutation_info', cand_data_list[ac5c_index_list[top_index]][0], peptide_feature2AA_seq([v for v in new_peptide_feature_list[ac5c_index_list[top_index]] if v != -2], AA_keys, ct_list, nt_list))
        for target_i in range(len(target_list)):
            target_index = target_list[target_i]
            target_name = data.keys()[target_index]
            print('  ', target_name, round(10**pred_y_list_list[target_i][ac5c_index_list[top_index]], 3), '(', round(pred_cov_list_list[target_i][ac5c_index_list[top_index]]**0.5,3), ')' )

    aib_ordered_total_PI_score_index = np.argsort(aib_total_pi_score_list)[::-1]
    for top_index in aib_ordered_total_PI_score_index[:3]:
        print( 'aib','total_pi_score', round(aib_total_pi_score_list[top_index],3), 'mutation_info', cand_data_list[aib_index_list[top_index]][0], peptide_feature2AA_seq([v for v in new_peptide_feature_list[aib_index_list[top_index]] if v != -2], AA_keys, ct_list, nt_list))
        for target_i in range(len(target_list)):
            target_index = target_list[target_i]
            target_name = data.keys()[target_index]
            print('  ', target_name, round(10**pred_y_list_list[target_i][aib_index_list[top_index]], 3), '(', round(pred_cov_list_list[target_i][aib_index_list[top_index]]**0.5,3), ')' )

    ac6c_ordered_total_PI_score_index = np.argsort(ac6c_total_pi_score_list)[::-1]
    for top_index in aib_ordered_total_PI_score_index[:3]:
        print( 'ac6c','total_pi_score', round(ac6c_total_pi_score_list[top_index],3), 'mutation_info', cand_data_list[ac6c_index_list[top_index]][0], peptide_feature2AA_seq([v for v in new_peptide_feature_list[ac6c_index_list[top_index]] if v != -2], AA_keys, ct_list, nt_list))
        for target_i in range(len(target_list)):
            target_index = target_list[target_i]
            target_name = data.keys()[target_index]
            print('  ', target_name, round(10**pred_y_list_list[target_i][ac6c_index_list[top_index]], 3), '(', round(pred_cov_list_list[target_i][ac6c_index_list[top_index]]**0.5,3), ')' )
            
else:
    ordered_total_PI_score_index = np.argsort(total_pi_score_list)[::-1]
    for top_index in ordered_total_PI_score_index[:10]:
        print('index', top_index, 'total_pi_score', round(total_pi_score_list[top_index],3), 'mutation_info', cand_data_list[top_index][0], peptide_feature2AA_seq([v for v in new_peptide_feature_list[top_index] if v != -2], AA_keys, ct_list, nt_list))
        for target_i in range(len(target_list)):
            target_index = target_list[target_i]
            target_name = data.keys()[target_index]
            print('  ', target_name, round(10**pred_y_list_list[target_i][top_index], 3), '(', round(pred_cov_list_list[target_i][top_index]**0.5,3), ')' )
  

