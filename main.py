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
import shap

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdCoordGen
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from sklearn import preprocessing 
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from acquisition_function import calc_PI_overfmax, calc_PI_underfmin
from feature_generator import calc_MorganCount, calc_mordred_descriptor
import metadata
from peptide_handler import peptide_feature2AA_seq, generate_new_peptitde
from smiles_handler import calc_smiles_skip_connection, replaceP_smiles, calc_smiles_woMC, calc_graph_connect,replaceS_smiles

def generate_peptide_from_mutation_info(input_aa_list, mutation_info):
    input_aa_list[2:4] = mutation_info[0]
    b  = copy.copy(input_aa_list)
    c = []
    for m_pos, m_aa in zip(mutation_info[1], mutation_info[2]):
        input_aa_list[4+m_pos] = m_aa
        c = copy.copy(input_aa_list)
    if b == c:
        return None
    else:
        new_peptide_smi, new_peptide_mol = generate_new_peptitde(base_index, input_aa_list, peptide_feature_list, smiles_list, AA_dict, AA_joint)  
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

data = pd.read_excel('./data/抗菌ペプチド情報_共同研究_pep9MDRP_20221017.xlsx')
data_num = 89
peptide_list = data['修正ペプチド配列'][:data_num]
smiles_list = data['SMILES'][:data_num]
mol_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]
AA_dict = metadata.AA_dict
AA_joint = metadata.AA_joint
AA_keys = list(AA_dict.keys())

link_index_list = []
for st in ['S5', 'R8', 's5', 'r8', '=']:
    link_index_list.append(AA_keys.index(st))

SR_index_list = []
for st in ['S', 'R', 's', 'r']:
    SR_index_list.append(AA_keys.index(st))

ct_list, nt_list = [], []
for peptide in peptide_list:
    peptide = unicodedata.normalize("NFKD", peptide).strip()
    ct,aa_list,nt = peptide.split('-')
    ct_list.append(ct)
    nt_list.append(nt)
ct_list = list(set(ct_list))
nt_list = list(set(nt_list))

peptide_feature_list = []
for peptide in peptide_list:
    peptide = unicodedata.normalize("NFKD", peptide).strip()
    ct,aa_list,nt = peptide.split('-')
    ct_index = ct_list.index(ct)
    nt_index = nt_list.index(nt)

    tmp_list = []
    for i, AA_key in enumerate(AA_dict.keys()):
        res = re.finditer(AA_key, aa_list)
        for s in res:
            tmp_list.append([s.span()[0], i])
    tmp_list = sorted(tmp_list, key=lambda x:float(x[0]))

    new_tmp_list = []
    for tmp in tmp_list:
        if tmp[0]+1 < len(aa_list):
            if tmp[1] in SR_index_list:
                if aa_list[tmp[0]+1] in ['5', '8']:
                    continue
        new_tmp_list.append(tmp)
    tmp_list = new_tmp_list

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
    peptide_feature_list.append(peptide_feature)

for i, pf in enumerate(peptide_feature_list):
    seq = peptide_feature2AA_seq(pf, AA_keys, ct_list, nt_list)

AA_keys = list(AA_dict.keys())

for i, pf in enumerate(peptide_feature_list):
    aa_seq = ''
    for j, k in enumerate(pf[4:]):
        if j in pf[2:4]:
            aa_seq += '='
        aa_seq += AA_keys[k]
  
    seq = ct_list[pf[0]]+'-'+aa_seq+'-'+nt_list[pf[1]]

max_len = np.max([len(v) for v in peptide_feature_list])
for peptide_feature in peptide_feature_list:
    pad_len = max_len - len(peptide_feature)
    peptide_feature += [-2] * pad_len

#Generate new peptides
#Select base sequence
#Correspnd to only L-amino acids

base_index = 8
input_aa_list = peptide_feature_list[base_index]
new_peptide_smi, new_peptide_mol = generate_new_peptitde(base_index, input_aa_list, peptide_feature_list, smiles_list, AA_dict, AA_joint)

smiles_woMC_list = []
for i in range(len(smiles_list)):
    seq_smi = calc_smiles_woMC(smiles_list[i], peptide_feature_list[i])
    smiles_woMC_list.append(seq_smi)

smiles_repP_list = []
for i in range(len(smiles_list)):
    seq_smi = replaceP_smiles(smiles_list[i], peptide_feature_list[i])
    smiles_repP_list.append(seq_smi)

#Calculation of Fingerprint, descriptor
descriptor_dimension = 1024
radial = 4
fp_proc_n = 4

with multiprocessing.Pool(processes = fp_proc_n) as pool:
  MACCS_fp = pool.starmap(mol2FP, [(mol, 'MACCS') for mol in mol_list])

with multiprocessing.Pool(processes = fp_proc_n) as pool:
  Morgan_r2_count = pool.starmap(mol2FP, [(mol, 'MorganCount', 2, descriptor_dimension) for mol in mol_list])

with multiprocessing.Pool(processes = fp_proc_n) as pool:
  Morgan_r4_count = pool.starmap(mol2FP, [(mol, 'MorganCount', 4, descriptor_dimension) for mol in mol_list])

mol_repP_skip7_list = [Chem.MolFromSmiles(calc_graph_connect(smi, peptide_feature, skip = 7)) for smi, peptide_feature in zip(smiles_repP_list, peptide_feature_list)]

with multiprocessing.Pool(processes = fp_proc_n) as pool:
  repP_skip7_MACCS_fp = pool.starmap(mol2FP, [(mol, 'MACCS') for mol in mol_repP_skip7_list])

with multiprocessing.Pool(processes = fp_proc_n) as pool:
  repP_skip7_Morgan_r2_count = pool.starmap(mol2FP, [(mol, 'MorganCount', 2, descriptor_dimension) for mol in mol_repP_skip7_list])

with multiprocessing.Pool(processes = fp_proc_n) as pool:
  repP_skip7_Morgan_r4_count = pool.starmap(mol2FP, [(mol, 'MorganCount', 4, descriptor_dimension) for mol in mol_repP_skip7_list])

#Preparation for building predicting models

def GP_predict(train_X, test_X, train_y, test_y):

    cov = physbo.gp.cov.gauss(train_X,ard = False)
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
    exp_list = data[target_name][:data_num]

    #Correction of mumerical data
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

    if smiles_type == 'original':
        if feature == 'MACCS':
            X = np.array(MACCS_fp)[filled_index_list]
        elif feature == 'Morgan_r2_count':
            X = np.array(Morgan_r2_count)[filled_index_list]
        elif feature == 'Morgan_r4_count':
            X = np.array(Morgan_r4_count)[filled_index_list]
    if smiles_type == 'smiles_repP_skip7':
        if feature == 'MACCS':
            X = np.array(repP_skip7_MACCS_fp)[filled_index_list]
        elif feature == 'Morgan_r2_count':
            X = np.array(repP_skip7_Morgan_r2_count)[filled_index_list]
        elif feature == 'Morgan_r4_count':
            X = np.array(repP_skip7_Morgan_r4_count)[filled_index_list]

    if model == 'physbo' and standardize: 
        ss = preprocessing.StandardScaler()
        X = ss.fit_transform(X)
  
    y = np.array(exp_modified_list)[filled_index_list]

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

            #SHAP analysis
            X_shap = X
            explainer = shap.TreeExplainer(model=train_model, feature_perturbation='tree_path_dependent')
            shap_values = explainer.shap_values(X=X_shap)
            shap.summary_plot(shap_values, X_shap, plot_type="bar")
            shap.plots.waterfall(explainer(X_shap)[8])

            y_pred = train_model.predict(X_test)
            y_train_pred = train_model.predict(X_train)

            y_pred_list += list(y_pred)
            y_test_list += list(y_test)

        elif model == 'physbo':
            [y_train_pred, y_train_pred_cov], [y_pred, y_pred_cov] = GP_predict(X_train, X_test, y_train, y_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_r = np.corrcoef(y_train, y_train_pred)[0][1]

            y_pred_list += list(y_pred)
            y_test_list += list(y_test)

        y_index_list += list(test_index)
        r = np.corrcoef(y_test_list, y_pred_list)[0][1]
        r2 = r2_score(y_test_list, y_pred_list)
        rmse = np.sqrt(mean_squared_error(y_test_list, y_pred_list))

    r = np.corrcoef(y_test_list, y_pred_list)[0][1]
    r2 = r2_score(y_test_list, y_pred_list)

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
    if target_name == 'Δ[θ] ([θ]222/[θ]208)':
        plt.savefig('./result/helix-like_feature'+feature+'_CV'+str(fold_n)+'_model'+model+'_smile'+smiles_type+'_scatter.png', dpi = 300)
    else:
        plt.savefig('./result/'+target_name+'_feature'+feature+'_CV'+str(fold_n)+'_model'+model+'_smile'+smiles_type+'_scatter.png', dpi = 300)
    plt.show()
    plt.clf()


#Validate predicition accuracy
#model list: 'RF', 'lightgbm'
#feature list: 'Morgan_r2', 'Morgan_r4','Morgan_r2_count', 'Morgan_r4_count', 'MACCS', 'Morgan_r2_MACCS', 'one-hot'
#fold_n: fold num of cross-validation

#target_index
#5: 'NZRC 3972', 6: 'DH5a', 7: 'Pseudomonas aeruginosa', 8: 'Staphylococcus aureus', 9: 'Proteus', 10: 'Staphylococcus epidermidis', 
#11: 'Proteus vulgaris', 12: 'Salmonella enterica subsp.', 13: 'Klebsiella pneumoniae', 14: 'MDRP', 15: 'Hemolysis', 16: 'Δ[θ] ([θ]222/[θ]208)'

target_index = 15
value_log = False
target_name = data.keys()[target_index]
exp_list = data[target_name][:data_num]

#Correction of mumerical data
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

if value_log == True:
    exp_modified_list = np.log10(exp_modified_list)
max = np.max(np.array(exp_modified_list)[filled_index_list][:,0])
min = np.min(np.array(exp_modified_list)[filled_index_list][:,0])

plt.hist([v[0] for v in np.array(exp_modified_list)[filled_index_list] if v[1] < 0], label = 'wo. linker', alpha = 0.6, bins = np.arange(min, max+0.01, (max-min)/20))
plt.hist([v[0] for v in np.array(exp_modified_list)[filled_index_list] if v[1] > 0], label = 'w. linker', alpha = 0.6, bins = np.arange(min, max+0.01, (max-min)/20))

plt.legend()
plt.title(target_name+' Log10='+str(value_log))
plt.xlabel(target_name)
plt.ylabel('frequency')
if target_name == 'Δ[θ] ([θ]222/[θ]208)':
    plt.savefig('./result/helix-like_dist_log'+str(value_log)+'.png', dpi = 300)
else:
    plt.savefig('./result/'+target_name+'_dist_log'+str(value_log)+'.png', dpi = 300)
plt.show()

#Recommend with BO
#Select base sequence
#Correspnd to only L-amino acids

base_index = 8
input_aa_list = copy.deepcopy(peptide_feature_list[base_index])

proc_n = 60
fp_proc_n = 4
mutation_num = 2
pep_len = len([v for v in input_aa_list[4:] if v >= 0])

NAA_index_list = list(range(21))
NNAA_index_list = [9,17,20,22,24,25,26]
mutatable_AA_index_list = NNAA_index_list
linker_index_list = [27, 28]
result_type = "worst500"

position_index_list = range(pep_len)
pos_comb_list = itertools.combinations(position_index_list, mutation_num)

mutation_info_list = [[[-1, -1], [], []]]

for pos_comb in pos_comb_list:
    for mutation_pos in pos_comb:
        for mutation_aa in itertools.product(mutatable_AA_index_list, repeat = mutation_num):
            mutation_info_list.append([[-1, -1], list(pos_comb), list(mutation_aa)])

#Generate information of peptides including staple 
linker_start_time = time.time()
linker_mutation_info_list = []
for m_i, mutation_info in enumerate(mutation_info_list):
  
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
linker_end_time = time.time()

new_peptide_mol_list1, new_peptide_smi_list1 = [], []
new_peptide_feature_list1 = []
cand_data_list1 = []

generate_start_time = time.time()

args_list = [(copy.deepcopy(peptide_feature_list[base_index]), mutation_info) for mutation_info in mutation_info_list]

with multiprocessing.Pool(processes = proc_n) as pool:
  new_peptide_data_list = pool.starmap(generate_peptide_from_mutation_info, args_list)
new_peptide_smi_list1 = [data[0] for data in new_peptide_data_list if data != None]
new_peptide_mol_list1 = [data[1] for data in new_peptide_data_list if data != None]
new_peptide_feature_list1 = [data[2] for data in new_peptide_data_list if data != None]
cand_data_list1 = [data[3] for data in new_peptide_data_list if data != None]

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
    seq_smi = replaceP_smiles(new_peptide_smi_list[i], new_peptide_feature_list[i])
    new_smiles_repP_list.append(seq_smi)

mol_list = new_peptide_mol_list

fp_start_time = time.time()
with multiprocessing.Pool(processes = fp_proc_n) as pool:
  Cand_Morgan_r2_fp = pool.starmap(mol2FP, [(mol, 'Morgan', 2, descriptor_dimension) for mol in mol_list])

with multiprocessing.Pool(processes = fp_proc_n) as pool:
  Cand_Morgan_r4_fp = pool.starmap(mol2FP, [(mol, 'Morgan', 4, descriptor_dimension) for mol in mol_list])

with multiprocessing.Pool(processes = fp_proc_n) as pool:
  Cand_MACCS_fp = pool.starmap(mol2FP, [(mol, 'MACCS') for mol in mol_list])

with multiprocessing.Pool(processes = fp_proc_n) as pool:
  Cand_Morgan_r2_count = pool.starmap(mol2FP, [(mol, 'MorganCount', 2, descriptor_dimension) for mol in mol_list])

with multiprocessing.Pool(processes = fp_proc_n) as pool:
  Cand_Morgan_r4_count = pool.starmap(mol2FP, [(mol, 'MorganCount', 4, descriptor_dimension) for mol in mol_list])

fp_end_time = time.time()

#smiles_repP_skip7
repP_start_time = time.time()
with multiprocessing.Pool(processes = proc_n) as pool:
  mol_repP_skip7_list = pool.starmap(smi2repP_skip, [(smi, peptide_feature, 7) for smi, peptide_feature in zip(new_smiles_repP_list, new_peptide_feature_list)])

with multiprocessing.Pool(processes = proc_n) as pool:
  Cand_repP_skip7_MACCS_fp = pool.starmap(mol2FP, [(mol, 'MACCS') for mol in mol_repP_skip7_list])

with multiprocessing.Pool(processes = proc_n) as pool:
  Cand_repP_skip7_Morgan_r2_count = pool.starmap(mol2FP, [(mol, 'MorganCount', 2, descriptor_dimension) for mol in mol_repP_skip7_list])

with multiprocessing.Pool(processes = proc_n) as pool:
  Cand_repP_skip7_Morgan_r4_count = pool.starmap(mol2FP, [(mol, 'MorganCount', 4, descriptor_dimension) for mol in mol_repP_skip7_list])

repP_end_time = time.time()

##target_index
#5: 'NZRC 3972', 6: 'DH5a', 7: 'Pseudomonas aeruginosa', 8: 'Staphylococcus aureus', 9: 'Proteus', 10: 'Staphylococcus epidermidis', 
#11: 'Proteus vulgaris', 12: 'Salmonella enterica subsp.', 13: 'Klebsiella pneumoniae', 14: 'MDRP', 15: 'Hemolysis', 16: 'Δ[θ] ([θ]222/[θ]208)'

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
    exp_list = data[target_name][:data_num]

    #Correction of mumerical data
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
    if value_log == True:
        exp_modified_list = np.log10(exp_modified_list)

    #Preparation to predict activities
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

    if model == 'physbo' and standardize:  
        ss = preprocessing.StandardScaler()
        X = ss.fit_transform(X)
        X_cand = ss.fit_transform(np.array(Cand_Morgan_r4_count))
    y = np.array(exp_modified_list)[filled_index_list]

    #Learning
    print('X', len(X), 'X_cand', len(X_cand), 'y', len(y))
    [y_pred_train, y_pred_cov_train], [y_pred, y_pred_cov] = GP_predict(X, X_cand, y, [0 for i in range(len(X_cand))])

    #calculate PI
    if threshold_list[target_i][0] == '<=':
        pi_list = calc_PI_underfmin(y_pred, y_pred_cov, np.log10(threshold_list[target_i][1]))
    elif threshold_list[target_i][0] == '>=':    
        pi_list = calc_PI_overfmax(y_pred, y_pred_cov, np.log10(threshold_list[target_i][1]))

    pred_y_list_list.append(y_pred)
    pred_cov_list_list.append(y_pred_cov)
    pi_list_list.append(pi_list)

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
with open('result/pred_y_list_list.pkl', mode='wb') as f:
    pickle.dump(pred_y_list_list, f) 
with open('result/pred_cov_list_list.pkl', mode='wb') as f:
    pickle.dump(pred_cov_list_list, f)   
    
   
if result_type == "NNAA":
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
            print('  ', target_name, round(10**pred_y_list_list[target_i][orn_index_list[top_index]], 3), '(', round(10**pred_cov_list_list[target_i][orn_index_list[top_index]]**0.5,3), ')' )

    dab_ordered_total_PI_score_index = np.argsort(dab_total_pi_score_list)[::-1]
    for top_index in dab_ordered_total_PI_score_index[:3]:
        print( 'dab','total_pi_score', round(dab_total_pi_score_list[top_index],3), 'mutation_info', cand_data_list[dab_index_list[top_index]][0], peptide_feature2AA_seq([v for v in new_peptide_feature_list[dab_index_list[top_index]] if v != -2], AA_keys, ct_list, nt_list))
        for target_i in range(len(target_list)):
            target_index = target_list[target_i]
            target_name = data.keys()[target_index]
            print('  ', target_name, round(10**pred_y_list_list[target_i][dab_index_list[top_index]], 3), '(', round(10**pred_cov_list_list[target_i][dab_index_list[top_index]]**0.5,3), ')' )
    
    ac5c_ordered_total_PI_score_index = np.argsort(ac5c_total_pi_score_list)[::-1]
    for top_index in ac5c_ordered_total_PI_score_index[:3]:
        print( 'ac5c','total_pi_score', round(ac5c_total_pi_score_list[top_index],3), 'mutation_info', cand_data_list[ac5c_index_list[top_index]][0], peptide_feature2AA_seq([v for v in new_peptide_feature_list[ac5c_index_list[top_index]] if v != -2], AA_keys, ct_list, nt_list))
        for target_i in range(len(target_list)):
            target_index = target_list[target_i]
            target_name = data.keys()[target_index]
            print('  ', target_name, round(10**pred_y_list_list[target_i][ac5c_index_list[top_index]], 3), '(', round(10**pred_cov_list_list[target_i][ac5c_index_list[top_index]]**0.5,3), ')' )

    aib_ordered_total_PI_score_index = np.argsort(aib_total_pi_score_list)[::-1]
    for top_index in aib_ordered_total_PI_score_index[:3]:
        print( 'aib','total_pi_score', round(aib_total_pi_score_list[top_index],3), 'mutation_info', cand_data_list[aib_index_list[top_index]][0], peptide_feature2AA_seq([v for v in new_peptide_feature_list[aib_index_list[top_index]] if v != -2], AA_keys, ct_list, nt_list))
        for target_i in range(len(target_list)):
            target_index = target_list[target_i]
            target_name = data.keys()[target_index]
            print('  ', target_name, round(10**pred_y_list_list[target_i][aib_index_list[top_index]], 3), '(', round(10**pred_cov_list_list[target_i][aib_index_list[top_index]]**0.5,3), ')' )

    ac6c_ordered_total_PI_score_index = np.argsort(ac6c_total_pi_score_list)[::-1]
    for top_index in aib_ordered_total_PI_score_index[:3]:
        print( 'ac6c','total_pi_score', round(ac6c_total_pi_score_list[top_index],3), 'mutation_info', cand_data_list[ac6c_index_list[top_index]][0], peptide_feature2AA_seq([v for v in new_peptide_feature_list[ac6c_index_list[top_index]] if v != -2], AA_keys, ct_list, nt_list))
        for target_i in range(len(target_list)):
            target_index = target_list[target_i]
            target_name = data.keys()[target_index]
            print('  ', target_name, round(10**pred_y_list_list[target_i][ac6c_index_list[top_index]], 3), '(', round(10**pred_cov_list_list[target_i][ac6c_index_list[top_index]]**0.5,3), ')' )
 
else:
    ordered_total_PI_score_index = np.argsort(total_pi_score_list)[::-1]
    Total_result_list = []
    for top_index in ordered_total_PI_score_index[:10]:
        print('index', top_index, 'total_pi_score', round(total_pi_score_list[top_index],3), 'mutation_info', cand_data_list[top_index][0], peptide_feature2AA_seq([v for v in new_peptide_feature_list[top_index] if v != -2], AA_keys, ct_list, nt_list))
        result_list = [peptide_feature2AA_seq([v for v in new_peptide_feature_list[top_index] if v != -2], AA_keys, ct_list, nt_list), round(total_pi_score_list[top_index],3)]
        for target_i in range(len(target_list)):
            target_index = target_list[target_i]
            target_name = data.keys()[target_index]
            print('  ', target_name, round(10**pred_y_list_list[target_i][top_index], 3), '(', round(10**pred_cov_list_list[target_i][top_index]**0.5,3), ')' )
            result_list.append(str(round(10**pred_y_list_list[target_i][top_index], 3)) + " " + '(' + str(round(10**pred_cov_list_list[target_i][top_index]**0.5,3)) + ')')
        Total_result_list.append(result_list)
    df = pd.DataFrame(Total_result_list)
    df.columns = ["Sequence","Score","NZRC 3972","DH5α","Pseudomonas aeruginosa","Staphylococcus aureus","Staphylococcus epidermidis","MDRP","Hemolysis"]
    df.to_csv("./result/top10.csv", encoding="shift_jis")
