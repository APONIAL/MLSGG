import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
# 相关系数法
advertising = pd.read_csv("../../data/advertising.csv")
print(advertising.head())
print(advertising.describe())
print(advertising.shape)
# axis=1表示操作行，把每一行的columns[0]删除
# 就等于删除第一列
# inplace=True 原地操作，不用再赋值
advertising.drop(advertising.columns[0], axis=1, inplace=True)
# 去掉空值
advertising.dropna(inplace=True)
# 提取特征和标签(目标值)
# 从 advertising 这个 pandas DataFrame 中删除名为 "Sales" 的列，
# 并将剩下的列赋值给变量 X
X = advertising.drop("Sales", axis=1)
# 取出Sales列并赋值给标签值y
y = advertising["Sales"]
print(X.shape)
print(y.shape)
# 计算皮尔逊相关系数
print(X.corrwith(y, method="pearson"))
# 使用corr计算advertising所有列之间的相关系数
corr_matrix = advertising.corr(method="pearson")
print(corr_matrix)
# 将相关系数矩阵画成热力图
import seaborn as sns

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature correlation Matrix")
plt.show()
