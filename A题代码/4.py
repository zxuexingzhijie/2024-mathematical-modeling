import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 从 Excel 文件中读取数据
file_path = r'C:\Users\周岩珏\Desktop\pythonProject1\第四题.xlsx'

# 假设 Excel 文件的第一张工作表（Sheet1）包含我们需要的数据
# 数据列假设为 "Received Signal Time" 和 "Received Signal Value"
data = pd.read_excel(file_path, sheet_name=0)

# 提取时间和信号值
time_data = data['Received Signal Time'].to_numpy()
signal_data = data['Received Signal Value'].to_numpy()

# 打印前几行数据以确认读取正确
print("读取的数据：")
print(data.head())

# 2. 间歇接收分析
# 计算时间差
time_diff = np.diff(time_data)  # 时间间隔
avg_time_diff = np.mean(time_diff)  # 平均时间间隔

# 打印时间间隔信息
print(f"时间间隔：{time_diff}")
print(f"平均时间间隔：{avg_time_diff}")

# 3. 使用FFT估计信号频率
# 我们将信号数据进行傅里叶变换以估计信号的频率成分
sampling_rate = 1 / avg_time_diff  # 采样频率，基于时间间隔的倒数
n = len(signal_data)  # 信号的长度
frequency = np.fft.fftfreq(n, d=avg_time_diff)  # 计算频率轴
fft_signal = np.fft.fft(signal_data)  # 傅里叶变换

# 计算功率谱（频域的幅度）
power_spectrum = np.abs(fft_signal) ** 2

# 4. 可视化结果
# 时域信号图
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time_data, signal_data, label="Received Signal", color='b')
plt.title("Time Domain: Received Signal")
plt.xlabel("Time (s)")
plt.ylabel("Signal Amplitude")
plt.grid(True)

# 频域信号图
plt.subplot(2, 1, 2)
plt.plot(frequency[:n//2], power_spectrum[:n//2], label="Power Spectrum", color='r')
plt.title("Frequency Domain: Power Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.grid(True)

plt.tight_layout()
plt.show()

# 5. 频率估计
# 频率估计可以通过找到功率谱的峰值频率来实现
peak_frequency_idx = np.argmax(power_spectrum[:n//2])  # 获取最大功率谱的索引
estimated_frequency = abs(frequency[peak_frequency_idx])  # 对应的频率值

print(f"估计的频率：{estimated_frequency} Hz")
