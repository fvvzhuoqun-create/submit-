import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim

# 1. 加载与预处理
df = pd.read_csv('cell_data_filtered_transposed.csv')
X = df.drop(columns=['Name']).values

# 缺失值填充与标准化
X_imputed = SimpleImputer(strategy='median').fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_imputed)

# 2. 定义 MLP 自编码器 (Autoencoder)
input_dim = X_scaled.shape[1]
encoding_dim = 64  # 降维后的目标维度

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# 3. 训练网络
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
X_tensor = torch.FloatTensor(X_scaled)

epochs = 150
for epoch in range(epochs):
    optimizer.zero_grad()
    encoded, decoded = model(X_tensor)
    loss = criterion(decoded, X_tensor)
    loss.backward()
    optimizer.step()

# 4. 提取降维特征
model.eval()
with torch.no_grad():
    latent_features, _ = model(X_tensor)
latent_np = latent_features.numpy()

# 5. 在新特征上进行聚类 (以 K=2 为例)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
final_labels = kmeans.fit_predict(latent_np)

# 保存数据
df.insert(1, 'MLP_Cluster', final_labels)
df.to_csv('cell_data_clustered_mlp.csv', index=False)