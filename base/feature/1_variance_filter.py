import numpy as np

# 构造特征
a = np.random.randn(100)
print(np.var(a))
b = np.random.normal(5, 0.1, size=100)
print(np.var(b))

# 构造特征向量（输入x）
# 将数组a和b垂直方向堆叠堆叠后形成两行，通过转置变成两列，把 (2, 100) 变成 (100, 2)。
X = np.vstack([a, b]).T
print(X)

# 低方差过滤
from sklearn.feature_selection import VarianceThreshold
# 低方差过滤：删除方差低于 0.01 的特征
var_thresh = VarianceThreshold(threshold=0.01)
X_filtered = var_thresh.fit_transform(X)
print(X_filtered)