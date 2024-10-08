import numpy as np  # 导入 numpy 库，用于科学计算
import matplotlib.pyplot as plt  # 导入 matplotlib 库，用于绘图
from mpl_toolkits.mplot3d import Axes3D  # 导入 Axes3D 库，用于 3D 绘图
from matplotlib.patches import Ellipse  # 从 matplotlib.patches 导入 Ellipse 类，用于绘制椭圆
import pandas as pd  # 导入 pandas 库，用于数据处理
from sklearn.cluster import KMeans  # 导入 KMeans 库，用于 K-Means 聚类
from sklearn.preprocessing import StandardScaler  # 导入 StandardScaler，用于数据标准化

# 加载 Air Quality 数据集
data = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',', header=0, na_values=-200)

# 选择多个特征（这里选择 CO(GT), PT08.S1(CO), NMHC(GT)）
features = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)']
data = data[features].dropna()  # 选择特征并去除缺失值

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# 设置聚类的分量数
num_clusters = 10

# 初始化 K-Means 模型
kmeans = KMeans(n_clusters=num_clusters, random_state=0)

# 拟合 K-Means 模型并进行聚类
kmeans.fit(X_scaled)
labels = kmeans.labels_  # 获取聚类标签
cluster_centers = kmeans.cluster_centers_  # 获取聚类中心

# 绘制 K-Means 聚类结果的 3D 图（选择前三个特征进行绘图）
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=labels, s=50, cmap='viridis', alpha=0.6)

# 绘制每个聚类中心的椭球体
for i in range(num_clusters):
    center = cluster_centers[i, :3]  # 聚类中心
    cov_matrix = np.cov(X_scaled[labels == i, :3].T)  # 计算协方差矩阵
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)  # 计算特征值和特征向量
    radii = 2 * np.sqrt(eigvals)  # 计算椭球体的半径

    # 绘制椭球体
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for j in range(len(u)):
        for k in range(len(v)):
            [x[j, k], y[j, k], z[j, k]] = np.dot([x[j, k], y[j, k], z[j, k]], eigvecs) + center

    ax.plot_wireframe(x, y, z, color='r', alpha=0.3)

ax.set_title("K-Means Clustering on Air Quality Data,by artist Guoruizhi220995111 ")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
plt.show()
