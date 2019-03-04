#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

'''参数初始化'''
T = 1 # GPS反馈周期
N = 80 / T
X = np.zeros((4, N), dtype=float)
X[:, 0] = [-100, 2, 200, 20]
Z = np.zeros((2, N), dtype=float)
Z[:, 0] = [X[0, 0], X[2, 0]]
w = 1e-2
Q = w * np.diag([0.5, 1, 0.5, 1])
R = 100 * np.eye(2)
F = np.array([[1, T, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, T],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

'''真实轨迹'''
for t in range(1, N):
    A = X[:, t-1]
    X[:, t] = F.dot(X[:, t-1])+ np.sqrt(Q).dot(np.random.randn(4, 1)).flatten()
    Z[:, t] = H.dot(X[:, t]) + np.transpose(np.sqrt(R).dot(np.random.randn(2, 1)))

Xkf = np.zeros((4, N), dtype=float)
Xkf[:, 0] = X[:, 0]
P = np.eye(4)

'''卡尔曼滤波'''
for i in range(1, N):
    # 预测(predict)

    Xpre = F.dot(Xkf[:, i-1])
    Ppre = (F.dot(P)).dot(np.transpose(F)) + Q

    # 更新(update)
    K = Ppre.dot(np.transpose(H)).dot(np.linalg.inv(H.dot(Ppre).dot(np.transpose(H)) + R))
    Xkf[:, i] = Xpre + K.dot(Z[:, i] - H.dot(Xpre))
    P = (np.eye(4) - K.dot(H)).dot(Ppre)

'''结果'''
plt.plot(X[0,:], X[2, :], 'g-', label="real")
plt.plot(Z[0,:], Z[1, :], 'b-s', label="measured")
plt.plot(Xkf[0, :], Xkf[2, :], 'r-+', label="filtered")
plt.xlabel("x/m")
plt.ylabel("y/m")
plt.title("Ship trajectory track")
plt.grid(True)
plt.legend(loc=('upper left'))
plt.show()


