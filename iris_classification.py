# 必要なライブラリのインポート
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
iris = load_iris()
X = iris.data
y = iris.target

# データフレームの作成（データの中身を確認しやすくするため）
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# データの確認
print("=== データの最初の5行 ===")
print(df.head())
print("\n=== データの基本統計量 ===")
print(df.describe())

# データの前処理
# 1. データの分割（訓練データとテストデータ）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. データの標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PyTorchのテンソルに変換
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# ニューラルネットワークモデルの定義
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # 入力層（4特徴量）→隠れ層（10ノード）
        self.fc2 = nn.Linear(10, 3)  # 隠れ層→出力層（3クラス）
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデルのインスタンス化
model = IrisNet()

# 損失関数と最適化手法の定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 学習の進捗を保存するリスト
losses = []

# モデルの学習
print("\n=== 学習開始 ===")
num_epochs = 100
for epoch in range(num_epochs):
    # フォワードパス
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # 損失値を記録
    losses.append(loss.item())
    
    # バックワードパスと最適化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 学習曲線の描画
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses, 'b-')
plt.title('学習曲線')
plt.xlabel('エポック数')
plt.ylabel('損失')
plt.grid(True)
plt.savefig('learning_curve.png')  # グラフを画像として保存
plt.close()

# モデルの評価
print("\n=== モデル評価 ===")
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs.data, 1)
    total = y_test_tensor.size(0)
    correct = (predicted == y_test_tensor).sum().item()
    print(f'テストデータでの精度: {100 * correct / total:.2f}%')

# 新しいデータでの予測例
print("\n=== 予測例 ===")
# テストデータの最初の3つのサンプルを使用
sample_data = X_test_tensor[:3]
with torch.no_grad():
    predictions = model(sample_data)
    _, predicted = torch.max(predictions.data, 1)
    
print("予測結果:")
for i in range(3):
    print(f"サンプル {i+1}: {iris.target_names[predicted[i]]}")

print("\n学習曲線が'learning_curve.png'として保存されました。")

# 特徴量の可視化
plt.figure(figsize=(15, 10))

# 特徴量ペアのプロット
sns.set_style("whitegrid")
sns.set_palette("husl")

# pairplotの作成
sns.pairplot(df, hue="target", diag_kind="hist")
plt.savefig('iris_features.png')
plt.close()

# 特徴量の相関行列のヒートマップ
plt.figure(figsize=(10, 8))
correlation = df.drop('target', axis=1).corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('特徴量の相関行列')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

print("\n特徴量の可視化が'iris_features.png'と'correlation_matrix.png'として保存されました。") 