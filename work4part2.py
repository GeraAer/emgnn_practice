import numpy as np  # 导入numpy库，用于科学计算
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot库，用于绘图
from matplotlib.patches import Ellipse  # 从matplotlib.patches模块导入Ellipse类，用于绘制椭圆
import pandas as pd  # 导入pandas库，用于数据处理
import re  # 导入正则表达式库，用于字符串处理

# 加载本地的 Old Faithful 数据集
data = pd.read_csv('old_faithful.csv')  # 读取本地的CSV文件

# 提取数值数据的函数
def extract_quantity(quantity_str):
    match = re.search(r"\[(\d*\.?\d+),", quantity_str)  # 使用正则表达式提取数值部分
    if match:
        return float(match.group(1))  # 返回提取到的数值并转换为浮点数
    return np.nan  # 如果未找到匹配项，返回NaN

# 应用提取函数到Duration和WaitingTime列
data['Duration'] = data['Duration'].apply(extract_quantity)  # 提取Duration列中的数值
data['WaitingTime'] = data['WaitingTime'].apply(extract_quantity)  # 提取WaitingTime列中的数值

# 确保数据集无缺失值
data = data.dropna()  # 删除包含NaN值的行

# 提取数据点
data = data[['Duration', 'WaitingTime']].values  # 提取喷发持续时间和等待时间两列

# 绘制原始数据
plt.figure(figsize=(8, 6))  # 创建绘图对象并设置大小
plt.scatter(data[:, 0], data[:, 1], s=10)  # 绘制数据点
plt.title("Old Faithful Geyser Data")  # 设置图形标题
plt.xlabel("Eruption duration (minutes)")  # 设置X轴标签
plt.ylabel("Waiting time (minutes)")  # 设置Y轴标签
plt.show()  # 显示绘图

# E步：计算后验概率
def e_step(data, means, covariances, weights):
    num_samples, num_features = data.shape  # 获取数据点的数量和特征数
    num_components = len(weights)  # 获取高斯混合模型的分量数
    responsibilities = np.zeros((num_samples, num_components))  # 初始化后验概率矩阵

    for k in range(num_components):  # 遍历每个高斯分布分量
        pdf = multivariate_gaussian(data, means[k], covariances[k])  # 计算每个数据点在第k个高斯分布下的概率密度
        responsibilities[:, k] = weights[k] * pdf  # 乘以对应的权重

    responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)  # 归一化后验概率
    return responsibilities  # 返回后验概率矩阵

# 计算多元高斯分布的概率密度函数
def multivariate_gaussian(x, mean, covariance):
    num_features = x.shape[1]  # 获取特征数
    diff = x - mean  # 计算数据点与均值的差值
    # 计算多元正态分布的概率密度
    return np.exp(-0.5 * np.sum(diff @ np.linalg.inv(covariance) * diff, axis=1)) / np.sqrt((2 * np.pi) ** num_features * np.linalg.det(covariance))

# M步：更新GMM参数
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

# EM算法进行GMM参数估计
def em_gmm(data, num_components, max_iter=100, tol=1e-4):
    num_samples, num_features = data.shape  # 获取数据点的数量和特征数

    # 初始化GMM参数
    means = data[np.random.choice(num_samples, num_components, replace=False)]  # 随机选择数据点作为初始均值
    covariances = np.array([np.cov(data, rowvar=False)] * num_components)  # 使用数据的协方差矩阵作为初始协方差矩阵
    weights = np.ones(num_components) / num_components  # 将初始权重设置为均匀分布

    log_likelihoods = []  # 用于存储对数似然值
    mean_trajectories = [means.copy()]  # 用于存储均值的迭代轨迹

    for i in range(max_iter):  # 迭代进行EM算法
        # E步
        responsibilities = e_step(data, means, covariances, weights)

        # M步
        means, covariances, weights = m_step(data, responsibilities)
        mean_trajectories.append(means.copy())  # 记录均值的变化轨迹

        # 计算对数似然
        log_likelihood = np.sum(np.log(np.sum([w * multivariate_gaussian(data, m, c) for w, m, c in zip(weights, means, covariances)], axis=0)))
        log_likelihoods.append(log_likelihood)

        # 检查收敛
        if i > 0 and abs(log_likelihood - log_likelihoods[-2]) < tol:
            break  # 如果对数似然值的变化小于阈值，则停止迭代

    return means, covariances, weights, log_likelihoods, mean_trajectories  # 返回估计的均值、协方差矩阵、权重、对数似然值和均值轨迹

# 绘制高斯分布的等高线
def plot_gaussian_ellipse(ax, mean, cov, color):
    v, w = np.linalg.eigh(cov)  # 计算协方差矩阵的特征值和特征向量
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # 计算椭圆的半长轴和半短轴
    u = w[0] / np.linalg.norm(w[0])  # 计算椭圆的旋转角度
    angle = np.arctan(u[1] / u[0])
    angle = 180.0 * angle / np.pi  # 将角度转换为度数
    ell = Ellipse(mean, v[0], v[1], angle=angle, color=color, alpha=0.5)  # 创建椭圆对象
    ax.add_patch(ell)  # 将椭圆添加到绘图中

# 执行EM算法进行GMM参数估计
num_components = 2  # 设置高斯混合模型的分量数为2
estimated_means, estimated_covariances, estimated_weights, log_likelihoods, mean_trajectories = em_gmm(data, num_components)

# 绘制数据点和均值轨迹
fig, ax = plt.subplots(figsize=(12, 8))  # 创建绘图对象并设置大小
scatter = ax.scatter(data[:, 0], data[:, 1], s=10, c='grey', alpha=0.5, label='Data points')  # 绘制数据点
colors = ['blue', 'orange']  # 设置颜色

for i, (mean, cov) in enumerate(zip(estimated_means, estimated_covariances)):  # 遍历每个高斯分布分量
    plot_gaussian_ellipse(ax, mean, cov, colors[i])  # 绘制高斯分布的等高线

for i, trajectory in enumerate(np.array(mean_trajectories).transpose(1, 0, 2)):  # 绘制均值的运动轨迹
    ax.plot(trajectory[:, 0], trajectory[:, 1], marker='o', linestyle='-', markersize=8, label=f'Mean trajectory {i+1}', color=colors[i])

# 优化图形外观
ax.set_title('GMM Mean Trajectories and Gaussian Ellipses for Old Faithful Data by Gridge a.k.a Guo Ruizhi a.k.a Cercis Morningstar', fontsize=16)  # 设置图形标题
ax.set_xlabel('Eruption duration (minutes)', fontsize=14)  # 设置X轴标签
ax.set_ylabel('Waiting time (minutes)', fontsize=14)  # 设置Y轴标签
ax.legend(fontsize=12)  # 设置图例字体大小
plt.grid(True)  # 显示网格线
plt.tight_layout()  # 自动调整子图参数以填充整个图形区域

plt.show()  # 显示绘图
