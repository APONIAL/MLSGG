import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

matplotlib.use('TkAgg')
# 定义数据集大小
n_samples = 1000
# 第1个主成分方向
# 使用 NumPy 从标准正态分布（均值为 0，标准差为 1）中随机采样 n_samples 个数。
component1 = np.random.normal(0, 1, n_samples)
# 第2个主成分方向
component2 = np.random.normal(0, 0.2, n_samples)
# 第3个方向（噪声，方差较小）
noise = np.random.normal(0, 0.1, n_samples)
# 构造3维数据
X = np.vstack([component1 - component2, component1 + component2, component2 + noise]).T

# 标准化
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# 应用PCA，将3维数据降维到2维
# 指定降维后的维数
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)
print(X_pca.shape)  # (1000, 2)

# 可视化
# 转换前的3维数据可视化
# plt.figure():是使用 Matplotlib 库创建一个新的图形对象（figure）
# figsize=(12, 4): 设置图形的大小，宽为12，高为4
fig = plt.figure(figsize=(12, 4))
# fig.add_subplot:在一个已有的 Matplotlib Figure 对象 fig 中添加一个三维子图（3D axes）
# 121 或 (1, 2, 1): 创建一个包含 1 行 2 列的子图网格，并选择第 1 个子图
# projection="3d": 指定该子图为 3D 投影类型，默认为 2D
ax1 = fig.add_subplot(121, projection="3d")
# ax1.scatter(...):调用 ax1（一个 3D 子图）的 .scatter() 方法，用于绘制三维散点图
# X[:, 0], X[:, 1], X[:, 2]: 获取 X 的第 1、2、3 列数据分别作为 X、Y、Z 坐标
# c="g": 设置散点颜色为绿色
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c="g")
ax1.set_title("Before PCA (3D)")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.set_zlabel("Feature 3")

# 转换后的2维数据可视化
ax2 = fig.add_subplot(122)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c="g")
ax2.set_title("After PCA (2D)")
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
plt.show()

# 手动构建线性相关的三组特征数据
n = 1000
# 定义两个主成分方向向量
pc1 = np.random.normal(0, 1, n)
pc2 = np.random.normal(0, 0.2, n)
# 定义不重要的第三主成分（噪声）
noise = np.random.normal(0, 0.05, n)
X_2 = np.vstack((pc1 + pc2, pc1 - pc2,  pc2 + noise)).T
print(X_2.shape)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_2)
print(X_pca.shape)
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(X_2[:, 0], X_2[:, 1], X_2[:, 2], c="g")
ax1.set_title("Before PCA (3D)")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")
ax1.set_zlabel("Feature 3")

# 转换后的2维数据可视化
ax2 = fig.add_subplot(122)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c="g")
ax2.set_title("After PCA (2D)")
ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
plt.show()
