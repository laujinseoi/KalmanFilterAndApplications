#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
设计一个卡尔曼滤波系统的过程如下：
1. 确定状态量、状态转移关系，建立转态转移矩阵
2. 确定观测值，建立观测矩阵，考虑随机扰动误差
3. 确定初始值
3. 代入卡尔曼滤波五大公式即可
下面是kalman滤波在自由落体运动目标跟踪中的例子
'''

import numpy as np
from matplotlib import pyplot as plt

N = 1000
Q = np.mat([[0, 0], [0, 0]])  # 过程噪声
R = 1   # 观测噪声方差
W = np.sqrt(Q) * np.mat(np.random.randn(2, N))    # 转态转移噪声
V = np.sqrt(R) * np.mat(np.random.randn(1, N))    # 观测噪声

# 系统矩阵
A = np.mat([[1,1],[0,1]])    # 状态转移矩阵
B = np.mat([[0.5],[1]])      # 控制矩阵
U = -10                      # 控制量
H = np.mat([1,0])            # 观测矩阵

# 初始化
X = np.mat(np.zeros((2, N), dtype=float))
X[:,0] = np.mat([95, 1]).T
P0 = np.mat([[10,0],[0,1]])
Z = np.mat(np.zeros((1,N),dtype=float))
Z[:,0] = H * X[:,0]
Xkf = np.mat(np.zeros((2,N), dtype=float))
Xkf[:,0] = X[:,0]
I = np.mat(np.eye(2))

# 迭代
for k in range(1, N):
    # 观测值
    X[:, k] = A * X[:, k-1] + U * B + W[:, k]
    Z[:, k] = H * X[:, k] + V[:, k]

    # 估计
    Xpre = A * Xkf[:, k-1] + B * U
    Ppre = A * P0 * A.T + Q

    # 更新
    K = Ppre * H.T * (H * Ppre * H.T + R).I
    Xkf[:, k] = Xpre + K * (Z[:, k] - H * Xpre)
    P0 = (I - K * H) * Ppre

plt.plot(range(0,N), np.array(Z[0,:])[0,:] - np.array(X[0,:])[0,:], '-r', label='Measure Err')
plt.plot(range(0,N), np.array(Xkf[0,:])[0,:] - np.array(X[0,:])[0,:], '-g', label='Kalman Err')
plt.xlabel('Sample Time/s')
plt.ylabel('Pos Err/m')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()
