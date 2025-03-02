import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import correlate

# 1. 读取 Excel 文件中的数据
file_path = r'C:\Users\***\Desktop\pythonProject1\第一问.xlsx'

# 假设数据存储在第一个工作表中，第一列是时间，第二列是接收到的信号值
data = pd.read_excel(file_path, sheet_name=0)

# 提取时间和接收到的信号值
time = data['Received Signal Time'].values
x_t = data['Received Signal Value'].values

# 2. 已知信号部分 (频率f0=30MHz，幅度A=4，阶段phi=45度)
A = 4
f0 = 30e6  # 30 MHz
phi = np.pi / 4  # 45 degrees

# 重建已知信号部分
x_signal = A * np.sin(2 * np.pi * f0 * time + phi)

# 3. 计算噪声部分 z(t) = x(t) - x_signal
z_t = x_t - x_signal

# 4. 噪声分析

# (a) 均值和方差
z_mean = np.mean(z_t)
z_variance = np.var(z_t)


# (b) 自相关函数
def autocorrelation(x):
    # 使用 `full` 模式得到完整的自相关函数
    corr = correlate(x, x, mode='full', method='auto')
    # 截取正时延部分，自相关函数在负时延部分是对称的
    return corr[len(x) - 1:]


z_autocorr = autocorrelation(z_t)

# 对自相关函数进行归一化，使得零延迟处的自相关值为 1
z_autocorr_normalized = z_autocorr / z_autocorr[0]

# (c) 功率谱密度
N = len(z_t)
dt = time[1] - time[0]  # 采样间隔
frequencies = fftfreq(N, dt)
z_fft = fft(z_t)
z_psd = np.abs(z_fft) ** 2 / N  # 标准化功率谱密度

# 5. 可视化

# 设置图形显示
fig, axs = plt.subplots(3, 2, figsize=(14, 12))

# (1) 绘制接收到的信号 x(t) 和已知信号 x_signal
axs[0, 0].plot(time, x_t, label='Received Signal x(t)', color='blue')
axs[0, 0].plot(time, x_signal, label='Known Signal x_signal', linestyle='--', color='orange')
axs[0, 0].set_title('Received Signal and Known Signal')
axs[0, 0].set_xlabel('Time [s]')
axs[0, 0].set_ylabel('Signal Amplitude')
axs[0, 0].legend()

# (2) 绘制噪声部分 z(t)
axs[0, 1].plot(time, z_t, label='Noise z(t)', color='red')
axs[0, 1].set_title('Noise z(t)')
axs[0, 1].set_xlabel('Time [s]')
axs[0, 1].set_ylabel('Noise Amplitude')

# (3) 绘制噪声自相关函数（归一化后的自相关函数）
axs[1, 0].plot(np.arange(0, len(z_autocorr_normalized)), z_autocorr_normalized, color='green')
axs[1, 0].set_title('Autocorrelation of Noise (Normalized)')
axs[1, 0].set_xlabel('Lag')
axs[1, 0].set_ylabel('Autocorrelation')

# (4) 绘制噪声的功率谱密度
axs[1, 1].plot(frequencies[:N // 2], z_psd[:N // 2], color='purple')
axs[1, 1].set_title('Power Spectral Density of Noise')
axs[1, 1].set_xlabel('Frequency [Hz]')
axs[1, 1].set_ylabel('Power')

# (5) 绘制接收到的信号 x(t) 和噪声部分 z(t)
axs[2, 0].plot(time, x_t, label='Received Signal x(t)', color='blue')
axs[2, 0].plot(time, z_t, label='Noise z(t)', color='red')
axs[2, 0].set_title('Received Signal and Noise z(t)')
axs[2, 0].set_xlabel('Time [s]')
axs[2, 0].set_ylabel('Amplitude')
axs[2, 0].legend()

# (6) 显示统计量（均值和方差）
axs[2, 1].axis('off')
text = f"Noise Mean: {z_mean:.3e}\nNoise Variance: {z_variance:.3e}"
axs[2, 1].text(0.1, 0.5, text, fontsize=12)

# 布局调整
plt.tight_layout()
plt.show()
