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
| `data`  | You can specify a dataset file. There are two default datasets, `Dataset_MODAN_initial.xlsx` and `Dataset_MODAN_initial_round1.xlsx`, in the "data" directory. |
| `AA_dict_update`  | You can register various amino acids used in MODAN. The items to be registered are a code, a name, and a type of amino acid used in MODAN.  For the type of amino acid, there are three types that can be registered. The first type is `a` as an α-amino acid. The second type is `a_a` as an α,α-disubstituted α-amino acid. The third type is `ring` as an α,α-disubstituted α-amino acid whose side chain is a cyclic structure. For α-aminoisobutyric acid, the code, the name, and the type are registered as `U`, `Aib`, and `a_a`, respectively. |
| `AA_joint_update` | You can register the code and SMILES strings of the side chain of the amino acid registered in `AA_dict_update`. For 'a_a', you need to register two types of SMILES strings of the side chain of the amino acid as a list format. For α-aminoisobutyric acid, the code and the two types of SMILES strings are registered as `U`, `C`, and `C`, respectively.|
| `base_atom` | You can select substituted atoms of α carbon to use Skip-7 representation. If you select phosphorus or sulfur, you set `P` and `S`, respectively.　|
| `Morgan_descriptor_dimension` | You can specify a descriptor dimension of a Morgan fingerprint. |
| `fold_n` | You can select a fold number to use cross-validation to validate surrogate models. |
| `value_log` |  You can choose whether or not to handle activity values in the common logarithms. If you choose the common logarithms, you set `True`. |
| `base_index` | You can specify the index number of a lead sequence in `data`. |
| `mutatable_AA_list` | You can specify the code of amino acids prepared for the substitution. |
| `mutation_num` | You can specify the number of times a lead sequence is converted into amino acids prepared for the substitution. |
| `smiles_select` | You can choose whether or not to select a SMILES representation used to construct a surrogate model yourself. If you select yourself, you set `True`. |
| `fingerprint_select` | You can choose whether or not to select a molecular fingerprint used to construct a surrogate model yourself. If you select yourself, you set `True`. |
|`target_list`| You can select a predicted item, a criterion, a SMILES representation, and a molecular fingerprint for each of the predicted items. For a criterion, you select the upper or lower bound as `<=` or `>=` besides a value. For a SMILES representation, if you select a standard representation　or a Skip-7 representation, you set `original` and `smiles_repP_skip7`, respectively. For a molecular fingerprint, if you select a MACCS fingerprint, a Morgan fingerprint (maximal radii is two), or a Morgan fingerprint (maximal radii is four), you set `MACCS`, `Morgan_r2_count`, and `Morgan_r4_count`, respectively. You do not need to select a SMILES representation and a molecular fingerprint, if you set `False` in `smiles_select` and `fingerprint_select`, respectively.|
| `result_type` | You can choose how a result is displayed. If you want to display a peptide with the highest total probability of improvement score including each amino acid prepared for the substitution, you set `Each_AA`. |
| `display_number` | You can choose the number to display a recommended peptide. |

## Contact
・Kei terayama (terayama@yokohama-cu.ac.jp)
