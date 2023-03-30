# MODAN

MODAN is a multi-objective Bayesian framework for the design of antimicrobial peptides (AMPs) considering various non-proteinogenic amino acids (NPAAs) containing α,α-disubstituted NPAAs and side-chain stapling.

## How to setup
```bash
pip install rdkit-pypi==2021.3.5 scikit-learn==1.1.0 pandas==1.4.2 matplotlib==3.5.2 pyyaml==6.0 physbo==1.0.1 mordred==1.2.0 openpyxl==3.0.10
```

## How to run MODAN

### 1. Clone this repository and move into it
```bash
git clone git@github.com:ycu-iil/MODAN
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
| `Data`  | You can specify a file of a dataset. There are two default dataset, in which `Dataset_MODAN_initial.xlsx` and `Dataset_MODAN_initial_round1.xlsx`. |
| `AA_dict_update`  | You can register various amino asids used in MODAN. The registering item are a code used in MODAN, amino acid’ name, and amino acid’ type. The total number of amino acid types that can be registered is three. The first type is 'a' as a α-amino acid. The second type is 'a_a' as α,α-disubstituted α-amino acid. The third type is 'ring' as α,α-disubstituted α-amino acid whose side chain is a cyclic structure.の中でも環構造になっているアミノ酸は'ring'|
| `AA_joint_update` | `AA_dict_update`で登録したアミノ酸のコードとアミノ酸の側鎖のSMILESを登録します。'a_a'はリスト形式で二種類のSMILESを登録します。 |
| `base_atom` | Skip-7 representation (ESI参照)を使用する際に、α carbonの変換原子を選択できます。　phosphorusなら`P`sulfurなら`S`　|
| `Morgan_descriptor_dimension` | Define descriptor dimension of the Morgan fingerprint |
| `fold_n` | cross-validation のフォールド数の指定ができます。 |
| `model` | 使用する機械学習手法を選べます。ガウス過程回帰なら`physbo`Light-GBMなら`lightgbm`Rondom forestなら　`RF` |
| `value_log` |  活性値を常用対数で扱うかどうかを選択できます。常用対数にするなら`True` |
| `base_index` | Define the index number of a lead sequence in `data`. |
| `mutatable_AA_list` | 置換アミノ酸のコードを指定します。 |
| `mutation_num` | リード配列に置換アミノ酸を置換させる数を指定します。 |
| `smiles_select` | ペプチド選出の際に使用する予測モデルのSMILES表記を手動で設定するかどうかを指定します。自身で選択する際は、`True` |
| `fingerprint_select` | ペプチド選出の際に使用する予測モデルのS特徴量を手動で設定するかどうかを指定します。自身で選択する際は、`True` |
|`target_list`| 予測する活性項目と活性も項目毎にcriterion、SMILE表記、特徴量を選択します。criterionはvalueの他に上限に設定するか下限に設定するかを選びます`>=`or`<=`。SMILE表記は、standard representationなら`original`、Skip-7 representationなら`smiles_repP_skip7`を選びます。特徴量は、MACCS fingerprintなら`MACCS`、Morgan fingerprint (maximal radii is two)なら`Morgan_r2_count`、Morgan fingerprint (maximal radii is four)なら`Morgan_r4_count`を選びます。 |
| `result_type` | 結果の表現方法を選択できます。もしdisplay top peptide including each amino acids prepared for the substitution置換アミノ酸毎に少なくとも1つ以上の変異が入ったTPIscore上位のペプチドを表示したい場合は`Each_AA` |

## Contact
・Kei terayama (terayama@yokohama-cu.ac.jp)
