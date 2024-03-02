import numpy as np
import matplotlib.pyplot as plt


# 与MATLAB代码的功能相同，用于生成和分析调制信号，并通过绘图进行可视化分析
# n为采样点数
def LFM(n):
    time_width = 10**(-4)   # 脉宽100微秒
    fre_width = 10**6       # 带宽1兆赫兹
    # A = 1                   # 振幅为1
    k = 2 * np.pi * fre_width / time_width   # 调制斜率
    Fs = 5 * 10**6          # 采样频率为5兆赫兹
    # N = int(time_width * Fs)  # 采样点数
    N = n

    n = np.linspace(-time_width / 2, time_width / 2, N)  # 生成时间序列
    # f = np.linspace(-Fs / 2, Fs / 2, N)  # 生成频率序列

    u = np.exp(1j * (k * n**2 / 2))  # 生成复数调制信号
    # Y = np.fft.fft(u)  # 对信号进行快速傅里叶变换

    # 归一化实部和虚部
    real_part_normalized = np.real(u) / np.max(np.abs(np.real(u)))
    imag_part_normalized = np.imag(u) / np.max(np.abs(np.imag(u)))

    # 拼接为复数向量
    normalized_complex_vector = np.vectorize(complex)(real_part_normalized, imag_part_normalized)

    return normalized_complex_vector

    # # 绘制时域图像
    # plt.figure()
    # plt.plot(n, np.real(u))
    # plt.title('Real Part of Modulated Signal vs Time')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    #
    # # 绘制频域图像
    # plt.figure()
    # plt.plot(n, np.imag(u))
    # plt.title('Imaginary Part of Modulated Signal vs Time')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    #
    # # 绘制频谱图
    # plt.figure()
    # plt.plot(f, np.fft.fftshift(np.abs(Y)))
    # plt.title('Frequency Spectrum of Modulated Signal')
    # plt.xlabel('Frequency')
    # plt.ylabel('Amplitude')
    #
    # plt.show()
