# NNpeptide_genration

アミノ酸記号 http://nomenclator.la.coocan.jp/chem/text/aminosym.htm
修正方針

- 基本はL体ということで,大文字はL体, 小文字はD体とする
- U=Aib (全体的に統一されていたのでOK)
- 一箇所(71番)Z=AibがあったのでZ->Uへ
- Z=Ac6c (全体的に統一されていたのでOK)
- X0=L-homoserine-(O-allyl) (No.11,12)
- X1 = Dab (No.45,51,52,61,62)
- X2 = Sarcosine (No.)
- 架橋している場合, アミノ酸文字の後に=を入れる
  - S5,S8の間には架橋が入る
- B=Ac5c (No.14)
- S5 = (S)-2-(4-pentenyl)Alanine
- R8 = (R)-2-(7-pentenyl)Alanine
- Orn (Orthinine?)が含まれている配列があるのでOに置き換え

特徴量への変換方針

2次構造予測スコアを特徴量に入れてしまう.

分割方法.

- まず'-'で分割. C末端, N末端は別処理. 
- AA配列の分
  - 数字が入っているか確認し, もし入っていたら数字入り要素を分割
  - 数字入り要素以外を1文字づつ分割
  - 用意したAAリストでindexを割り振る
- 架橋は別フラグを用意する.
- 特徴量の全体は, C末端情報, N末端情報, AA配列情報(indexの後に架橋しているかのbit,D体(0)L体(1)のbit, の繰り返し) 

## Requirements

### How to setup

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

`./data` に `抗菌ペプチド情報_共同研究(寺山先生)_出水_修正版20220322.xlsx` を配置する。
番号83は空行なので削除。
