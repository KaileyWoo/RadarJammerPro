from abc import ABC
from gymnasium import Env, spaces
import numpy as np
from LFM.LFM import LFM

dimension = 16  # 发射波形码长
s_dimension = (dimension * 2, 1)
j_dimension = (dimension, dimension * 2)


# 将复数矩阵转为实数矩阵
def complex_to_matrix(complex_matrix):
    # 将复数矩阵拆解为实部和虚部，并展平为一维向量
    real_part = np.real(complex_matrix).flatten()
    imag_part = np.imag(complex_matrix).flatten()
    # 将实部和虚部拼接在一起
    real_vector = np.concatenate((real_part, imag_part))
    return real_vector


# 将实数矩阵转为复数矩阵
def matrix_to_complex(real_vector, original_shape):
    # 将实数向量重新分割为实部和虚部的一维向量
    n = len(real_vector) // 2
    real_part = real_vector[:n]
    imag_part = real_vector[n:]
    # 将实部和虚部的一维向量重新形状为原始复数矩阵的形状
    real_matrix = np.reshape(real_part, original_shape)
    imag_matrix = np.reshape(imag_part, original_shape)
    # 将实部和虚部重新组合为复数矩阵
    complex_matrix = real_matrix + 1j * imag_matrix
    return complex_matrix


def normalize_complex_matrix(complex_matrix):
    # 提取实部和虚部
    real_part = np.real(complex_matrix)
    imag_part = np.imag(complex_matrix)
    # 对实部和虚部分别进行归一化
    real_part_normalized = (real_part - np.min(real_part)) / (np.max(real_part) - np.min(real_part)) * 2 - 1
    imag_part_normalized = (imag_part - np.min(imag_part)) / (np.max(imag_part) - np.min(imag_part)) * 2 - 1
    # 重新组合为归一化后的复数矩阵
    normalized_complex_matrix = real_part_normalized + 1j * imag_part_normalized
    return normalized_complex_matrix


# 雷达环境
class RadarEnv(Env, ABC):
    def __init__(self, radar_shared_data, jammer_shared_data):
        super().__init__()
        self.radar_shared_data = radar_shared_data
        self.jammer_shared_data = jammer_shared_data

        # 输出
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=s_dimension, dtype=np.float32)
        # 输入
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=j_dimension, dtype=np.float32)

        self.observation = []
        self.start_train_flag = True

    def step(self, action):
        terminated = True
        truncated = False
        reward = 0

        # 转换为复数
        s = matrix_to_complex(action, (dimension, 1))
        s_H = np.conj(s).T
        # 发射波形
        self.radar_shared_data.put({'s': s})

        # 获取干扰机回波相关信息
        data = self.jammer_shared_data.get()  # 若为空，进程会被阻塞等待
        data_a = data['a']  # 目标回波幅度
        data_r_s = data['R(s)']  # 干扰协方差矩阵
        data_r_s = normalize_complex_matrix(data_r_s)  # 归一化
        self.observation = complex_to_matrix(data_r_s).reshape(j_dimension)

        # 计算复数矩阵的逆
        data_r_s_inverse = np.linalg.pinv(data_r_s)
        # 计算接受滤波器 w
        w = data_r_s_inverse @ s
        w_H = np.conj(w).T

        # 计算 |w^H s|^2
        result = np.abs(w_H @ s) ** 2

        # 最大化 SJNR
        SJNR = result / (w_H @ data_r_s @ w)
        reward_SJNR = np.abs(SJNR)
        reward += reward_SJNR

        # 约束奖励 || s-s0 || <= η (Eta)
        s0 = LFM(dimension)
        ss = s - s0
        nn = np.linalg.norm(ss)
        if np.linalg.norm(s - s0) > 1:
            reward -= 1

        # 约束奖励 s^H s = 1
        if s_H @ s != 1:
            reward -= 1

        return self.observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.start_train_flag is True:
            self.start_train_flag = False
            s = LFM(dimension).reshape((dimension, 1))
            self.radar_shared_data.put({'s': s})   # 发射波形
            # 获取干扰机回波相关信息
            data = self.jammer_shared_data.get()  # 若为空，进程会被阻塞等待
            data_a = data['a']  # 目标回波幅度
            data_r_s = data['R(s)']  # 干扰协方差矩阵
            data_r_s = normalize_complex_matrix(data_r_s)  # 归一化
            self.observation = complex_to_matrix(data_r_s).reshape(j_dimension)

        return self.observation, {}

    @staticmethod
    def get_hermite():
        # 生成对角部分
        a = np.random.uniform(-1, 1, size=(dimension, dimension))
        a = np.diag(np.diag(a))  # 只保留对角元素，其余元素置为零
        # 生成非对角部分
        u1 = np.random.randn(dimension, dimension)
        u2 = np.random.randn(dimension, dimension)
        b_real = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)  # 生成一个服从标准正态分布的实数矩阵，作为复数矩阵的实部
        b_imag = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)  # 生成一个服从标准正态分布的实数矩阵，作为复数矩阵的虚部
        b = b_real + 1j * b_imag  # 生成一个复数矩阵
        # 生成Hermite矩阵
        hermite = a + b + np.conj(b).transpose()  # 将 A 和 B 组合成一个 Hermite 矩阵，即 C = A + B + B^H

        return hermite


# 干扰机环境
class JammerEnv(Env, ABC):
    def __init__(self, radar_shared_data, jammer_shared_data):
        super().__init__()
        self.radar_shared_data = radar_shared_data
        self.jammer_shared_data = jammer_shared_data

        # 输出
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=j_dimension, dtype=np.float32)
        # 输入
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=s_dimension, dtype=np.float32)

        self.reset_flag = False
        self.reset_s = []

        self.observation = []
        self.start_train_flag = True

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0

        if self.reset_flag is True:
            self.reset_flag = False
            s = self.reset_s
        else:
            # 接收波形
            data = self.radar_shared_data.get()  # 若为空，进程会被阻塞等待
            s = data['s']  # 接受雷达发射波形
            terminated = True

        s_H = np.conj(s).T
        self.observation = complex_to_matrix(s).reshape(s_dimension)

        # 生成一个随机数在[-1, 1]之间的 目标回波幅度
        a = np.random.uniform(low=-1, high=1)

        # 创建单位矩阵
        identity_matrix = np.eye(dimension)

        # 定义噪声能量，生成均值为0、标准差为1的高斯噪声
        noise = np.random.normal(0, 1, 1000)
        # 计算噪声的能量（均方值）
        noise_energy = np.mean(noise ** 2)

        # 存储转发转移矩阵 Jm
        # 转换为复数
        J = matrix_to_complex(action, (dimension, dimension))

        # 计算干扰协方差矩阵 R(s)
        r_s = a ** 2 * J @ s @ s_H @ J.T + noise_energy * identity_matrix

        # 发送数据给雷达
        self.jammer_shared_data.put({'a': a, 'R(s)': r_s})

        # 计算复数矩阵的逆
        r_s_inverse = np.linalg.pinv(r_s)
        # 计算接受滤波器 w
        w = r_s_inverse @ s
        w_H = np.conj(w).T

        # 计算 |w^H s|^2
        result = np.abs(w_H @ s) ** 2

        # 最小化 SJNR
        SJNR = result / (w_H @ r_s @ w)
        reward_SJNR = - np.abs(SJNR)
        reward += reward_SJNR

        return self.observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.reset_flag = True

        # 接收波形
        data = self.radar_shared_data.get()  # 若为空，进程会被阻塞等待
        self.reset_s = data['s']  # 接受雷达发射波形
        self.observation = complex_to_matrix(self.reset_s).reshape(s_dimension)

        return self.observation, {}

