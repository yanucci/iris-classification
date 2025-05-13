# アヤメ品種分類 - ニューラルネットワーク練習

このプロジェクトは、有名なIrisデータセットを使用して、アヤメの品種を分類するニューラルネットワークモデルを実装したものです。

## 機能

- PyTorchを使用したニューラルネットワークの実装
- アヤメの4つの特徴量（がく片の長さ・幅、花弁の長さ・幅）から品種を予測
- データの可視化（学習曲線、特徴量の分布、相関行列）
- 高精度な分類（テストデータでの精度: 約97%）

## 可視化結果

プロジェクトでは以下の3つの可視化を生成します：

1. `learning_curve.png`: モデルの学習過程を示す学習曲線
2. `iris_features.png`: 特徴量のペアプロット（品種ごとの分布）
3. `correlation_matrix.png`: 特徴量間の相関行列

## 必要なライブラリ

- numpy
- pandas
- scikit-learn
- torch (PyTorch)
- matplotlib
- seaborn

## 使用方法

1. 必要なライブラリをインストール：
```bash
pip install numpy pandas scikit-learn torch matplotlib seaborn
```

2. プログラムを実行：
```bash
python iris_classification.py
```

## モデル構造

- 入力層: 4ノード（4つの特徴量）
- 隠れ層: 10ノード（ReLU活性化関数）
- 出力層: 3ノード（3つの品種）
- 最適化手法: Adam
- 損失関数: CrossEntropyLoss

## 学習パラメータ

- エポック数: 100
- 学習率: 0.01
- バッチサイズ: 全データ
- テストデータ割合: 20% 