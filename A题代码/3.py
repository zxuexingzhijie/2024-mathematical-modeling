import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 从Excel文件中读取数据
file_path = r'C:\Users\周岩珏\Desktop\pythonProject1\第三题.xlsx'
data = pd.read_excel(file_path)

# 2. 提取时间和信号值数据
time = data['Received Signal Time'].values
signal = data['Received Signal Value'].values

# 3. 获取采样间隔
T_s = time[1] - time[0]  # 计算时间间隔

# 4. 进行零填充，使得FFT的分辨率更高
N = len(signal)  # 信号长度
N_padded = 2**np.ceil(np.log2(N)).astype(int)  # 选择大于等于N的最小2的幂次，增加FFT的分辨率
signal_padded = np.pad(signal, (0, N_padded - N), 'constant', constant_values=(0, 0))

# 5. 对信号进行快速傅里叶变换（FFT）
fft_signal = np.fft.fft(signal_padded)  # 计算FFT
frequencies = np.fft.fftfreq(N_padded, T_s)  # 生成频率轴

# 6. 计算频谱的幅度（只考虑正频率部分）
magnitude = np.abs(fft_signal[:N_padded // 2])
frequencies = frequencies[:N_padded // 2]  # 只取正频率部分

# 7. 对频谱进行平滑处理，减小噪声对频谱的影响
def smooth_spectrum(magnitude, window_size=5):
    """使用滑动平均法对频谱进行平滑处理"""
    return np.convolve(magnitude, np.ones(window_size)/window_size, mode='same')

# 平滑后的频谱
smoothed_magnitude = smooth_spectrum(magnitude)

# 8. 使用对数尺度来展示频域数据（对于幅度的对数）
log_magnitude = np.log(smoothed_magnitude + 1e-10)  # 防止出现log(0)的情况

# 9. 找到主峰对应的频率
peak_frequency = frequencies[np.argmax(smoothed_magnitude)]  # 找到幅度最大值对应的频率

# 输出估计的频率
print(f"估计的信号频率：{peak_frequency:.2e} Hz")

# 10. 可视化时域信号和频域信号
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# 时域信号
axs[0].plot(time, signal, label='Received Signal', color='b')
axs[0].set_title('Received Signal in Time Domain')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Signal Value')
axs[0].grid(True)

# 频域信号（使用对数尺度）
axs[1].plot(frequencies, log_magnitude, label='Log Magnitude of Smoothed FFT', color='r')
axs[1].set_title('Signal Frequency Spectrum (Log Scale)')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Log Magnitude')
axs[1].grid(True)

# 显示估计的频率
axs[1].axvline(x=peak_frequency, color='g', linestyle='--', label=f'Estimated Frequency: {peak_frequency:.2e} Hz')
axs[1].legend()

# 显示图形
plt.tight_layout()
plt.show()
