# MODAN

MODAN is a multi-objective Bayesian framework for the design of antimicrobial peptides (AMPs) considering various non-proteinogenic amino acids (NPAAs) containing α,α-disubstituted NPAAs and side-chain stapling.

## Requirements
これは全部かく?

## How to setup

```bash
pip install rdkit-pypi
pip install scikit-learn
pip install lightgbm
pip install optuna
pip install pandas
pip install matplotlib
pip install mordred
pip install physbo
```

## How to run MODAN

### 1.Clone this repository and move into it
```bash
git clone git@github.com/ycu-iil/MODAN
cd MODAN
```

### 2. Recommend AMP candidates

```bash
python main.py
```

### 3. Advaced usage

- Dataset
  - Dataset_MODAN_initial.xlsx
  - Dataset_MODAN_initial_round1.xlsx
  
- NPAAs
  - 'amino acid code': - 'amino acid name' - 'amino acid type'
    - amino acid type: 'a' is α-amino acid, 'a_a' is a α,α-disubstituted α-amino acid, 'ring' is a cyclic amino acid 

- A atom changed from α carbon 
  - P (phosphorus)
  - S (sulfur)
  
- A criterion each target
  - '<=' or '>='
  - criterion value

- Paralell processing
  - proc_n:
  - fp_proc_n:
  
- Number of mutaion

- Amino acids prepared for the substitution

- The way of output
  - None (standard)
  - Each_AA (display top peptide including each amino acids prepared for the substitution)
  
- Availability of logarithm

- Availability of standardization

- The type of SMILES
  - 'original'
  - 'smiles_repP_skip7'

- The type of a fingerprint
  - 'MACCS'
  - 'Morgan_r2_count'
  - 'Morgan_r4_count'

- Availability of standardization

## Contact
・Kei terayama (terayama@yokohama-cu.ac.jp)
