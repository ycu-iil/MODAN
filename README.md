# MODAN

MODAN is a multi-objective Bayesian framework for the design of antimicrobial peptides (AMPs) considering various non-proteinogenic amino acids (NPAAs) containing α,α-disubstituted NPAAs and side-chain stapling.

## Confirmed
Python: 3.9.12

## How to setup
```bash
pip install rdkit-pypi==2021.3.5 scikit-learn==1.1.0 pandas==1.4.2 matplotlib==3.5.2 pyyaml==6.0 physbo==1.0.1 mordred==1.2.0 openpyxl==3.0.10
```

## How to run MODAN

### 1. Clone this repository and move into it
```bash
git clone git@github.com:ycu-iil/MODAN.git
cd MODAN
```

### 2. Prepare a config file
The Recommndataion of options are described in the "Advaced usage" section. For details, please refer to a sample file ("config/setting.yaml").

### 3. Recommend AMP candidates 
```bash
python main.py -c config/setting_paper.yaml
```

### 4. Advanced usage

| Option  | Description |
| ------------- | ------------- |
| `data`  | You can specify a file of a dataset. There are two default dataset, in which `Dataset_MODAN_initial.xlsx` and `Dataset_MODAN_initial_round1.xlsx`. |
| `AA_dict_update`  | You can register various amino asids used in MODAN. The registering item are a code used in MODAN, amino acid’ name, and amino acid’ type. The total number of amino acid types that can be registered is three. The first type is 'a' as a α-amino acid. The second type is 'a_a' as α,α-disubstituted α-amino acid. The third type is 'ring' as α,α-disubstituted α-amino acid whose side chain is a cyclic structure. For α-aminoisobutyric acid, the code, the amino acid’ name, and the amino acid’ type are registered as `U`, `Aib`, and `a_a`, respectively. |
| `AA_joint_update` | You can register the code and SMILES of the side chain of the amino acid registered at `AA_dict_update`. As for 'a_a', you need to register two types of SMILES of the side chain of the amino acid as a list format.　For α-aminoisobutyric acid, the code and the two types of SMILES are registerd as `U`, `C`, and `C`, respectively.|
| `base_atom` | If you can use Skip-7 representation (reffer to ESI)どうするか, you can choose substituted atoms of α carbon. If you choose phosphorus, you set `P`. If you choose sulfur, you set `S`.　|
| `Morgan_descriptor_dimension` | Define descriptor dimension of the Morgan fingerprint |
| `fold_n` | You can choose fold number used at cross-validation to validate prediciton models. |
| `value_log` |  You can choose whether to handle activity values in the common logarithms or not. If you choose the common logarithms. you set `True`. |
| `base_index` | Define the index number of a lead sequence in `data`. |
| `mutatable_AA_list` | You can specify the code of amino acids prepared for the substitution. |
| `mutation_num` | You can specify the number that a lead sequence is converted into amino acids prepared for the substitution. |
| `smiles_select` | You can choose whether to choose a SMIELS representation used to construct a surrogate model yourself or not. If you choose this yourself, you set `True`. |
| `fingerprint_select` | You can choose whether to choose a molecular fignerprint used to construct a surrogate model yourself or not. If you choose this yourself, you set `True`. |
|`target_list`| You can select predicted items, a criterion, a SMILES represenataion, and a molecular fingerprint each the predicted items. As for a criterion, you choose the upper or lower limit as `<=`or`>=` besides a valueこのあたり不安. As for a SMILES representation, if you choose a standard representation, you set `original`. If you choose a Skip-7 representation, you set `smiles_repP_skip7`. As for a molecular fingerprint, if you choose a MACCS fingerprint, you set `MACCS`. If you choose a Morgan fingerprint (maximal radii is two), you set `Morgan_r2_count`. If you choose a Morgan fingerprint (maximal radii is four), you set `Morgan_r4_count`. You do not need to select a SMILES represenataion and a molecular fingerprint, if you set `False` at `smiles_select` and `fingerprint_select`, respectively. |
| `result_type` | You can choose how to output of a result. If you want to display top peptide including each amino acids prepared for the substitution, you set `Each_AA`. |
| `display_number` | You can choose the number to display a recommended candidate peptide. |

## Contact
・Kei terayama (terayama@yokohama-cu.ac.jp)
