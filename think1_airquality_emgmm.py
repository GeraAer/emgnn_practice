import numpy as np  # 导入 numpy 库，用于科学计算
import matplotlib.pyplot as plt  # 导入 matplotlib 库，用于绘图
from mpl_toolkits.mplot3d import Axes3D  # 导入 Axes3D 库，用于 3D 绘图
import pandas as pd  # 导入 pandas 库，用于数据处理
from sklearn.mixture import GaussianMixture  # 导入 GaussianMixture，用于 GMM 聚类
from sklearn.preprocessing import StandardScaler  # 导入 StandardScaler，用于数据标准化

# 加载 Air Quality 数据集
data = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',', header=0, na_values=-200)

# 选择多个特征（这里选择 CO(GT), PT08.S1(CO), NMHC(GT)）
features = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)']
data = data[features].dropna()  # 选择特征并去除缺失值

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# 绘制原始数据的 3D 图（选择前三个特征进行绘图）
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c='grey', s=50)
ax.set_title("Original Air Quality Data")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
plt.show()

# E 步：计算后验概率
def e_step(data, means, covariances, weights):
    num_samples, num_features = data.shape  # 获取数据点的数量和特征数
    num_components = len(weights)  # 获取高斯混合模型的分量数
    responsibilities = np.zeros((num_samples, num_components))  # 初始化后验概率矩阵

    for k in range(num_components):  # 遍历每个高斯分布分量
        pdf = multivariate_gaussian(data, means[k], covariances[k])  # 计算每个数据点在第 k 个高斯分布下的概率密度
        responsibilities[:, k] = weights[k] * pdf  # 乘以对应的权重

    responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)  # 归一化后验概率
    return responsibilities  # 返回后验概率矩阵

# 计算多元高斯分布的概率密度函数
def multivariate_gaussian(x, mean, covariance):
    num_features = x.shape[1]  # 获取特征数
    diff = x - mean  # 计算数据点与均值的差值
    # 计算多元正态分布的概率密度
    return np.exp(-0.5 * np.sum(diff @ np.linalg.inv(covariance) * diff, axis=1)) / np.sqrt((2 * np.pi) ** num_features * np.linalg.det(covariance))

# M 步：更新 GMM 参数
def m_step(data, responsibilities):
    num_samples, num_features = data.shape  # 获取数据点的数量和特征数
    num_components = responsibilities.shape[1]  # 获取高斯混合模型的分量数

    weights = responsibilities.sum(axis=0) / num_samples  # 更新权重
    means = np.dot(responsibilities.T, data) / responsibilities.sum(axis=0)[:, np.newaxis]  # 更新均值

    covariances = np.zeros((num_components, num_features, num_features))  # 初始化协方差矩阵
    for k in range(num_components):  # 遍历每个高斯分布分量
        diff = data - means[k]  # 计算数据点与均值的差值
        covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / responsibilities[:, k].sum()  # 更新协方差矩阵

    return means, covariances, weights  # 返回更新后的均值、协方差矩阵和权重

# EM 算法进行 GMM 参数估计
def em_gmm(data, num_components, max_iter=100, tol=1e-4):
    num_samples, num_features = data.shape  # 获取数据点的数量和特征数

    # 初始化 GMM 参数
    means = data[np.random.choice(num_samples, num_components, replace=False)]  # 随机选择数据点作为初始均值
    covariances = np.array([np.cov(data, rowvar=False)] * num_components)  # 使用数据的协方差矩阵作为初始协方差矩阵
    weights = np.ones(num_components) / num_components  # 将初始权重设置为均匀分布

    log_likelihoods = []  # 用于存储对数似然值
    mean_trajectories = [means.copy()]  # 用于存储均值的迭代轨迹

    for i in range(max_iter):  # 迭代进行 EM 算法
        # E 步
        responsibilities = e_step(data, means, covariances, weights)

        # M 步
        means, covariances, weights = m_step(data, responsibilities)
        mean_trajectories.append(means.copy())  # 记录均值的变化轨迹

        # 计算对数似然
        log_likelihood = np.sum(np.log(np.sum([w * multivariate_gaussian(data, m, c) for w, m, c in zip(weights, means, covariances)], axis=0)))
        log_likelihoods.append(log_likelihood)

        # 检查收敛
        if i > 0 and abs(log_likelihood - log_likelihoods[-2]) < tol:
            break  # 如果对数似然值的变化小于阈值，则停止迭代

    return means, covariances, weights, log_likelihoods, mean_trajectories  # 返回估计的均值、协方差矩阵、权重、对数似然值和均值轨迹

# 执行 EM 算法进行 GMM 参数估计
num_components = 10  # 设置高斯混合模型的分量数为 10
estimated_means, estimated_covariances, estimated_weights, log_likelihoods, mean_trajectories = em_gmm(X_scaled, num_components)

# 绘制数据点和均值轨迹
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c='grey', s=50, label='Data points')
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# 绘制均值的运动轨迹
for i, trajectory in enumerate(np.array(mean_trajectories).transpose(1, 0, 2)):
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], marker='o', linestyle='-', markersize=8, label=f'Mean trajectory {i+1}', color=colors[i])

ax.set_title('GMM Mean Trajectories for Air Quality Data', fontsize=16)
ax.set_xlabel('Feature 1', fontsize=14)
ax.set_ylabel('Feature 2', fontsize=14)
ax.set_zlabel('Feature 3', fontsize=14)
ax.legend(fontsize=12)
plt.show()
