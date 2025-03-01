import pandas as pd
import numpy as np
import random
from scipy.fftpack import fft
from scipy.signal import butter, lfilter

# 步骤1：加载并过滤数据
# 加载飞行阶段3的Excel文件
file_path = r'C:\Users\周岩珏\Desktop\pythonProject1\第三题.xlsx'
data = pd.read_excel(file_path)

# 提取时间和信号值
time_filtered = data['Received Signal Time'].values
received_signal_filtered = data['Received Signal Value'].values

# 步骤2：使用FFT进行频谱分析
# 使用快速傅里叶变换（FFT）获取频率谱
n = len(received_signal_filtered)
sampling_interval = time_filtered[1] - time_filtered[0]  # 假设时间值均匀分布
sampling_frequency = 1 / sampling_interval  # 采样频率

# 进行FFT变换
fft_values = fft(received_signal_filtered)
fft_frequencies = np.fft.fftfreq(n, sampling_interval)

# 找到频谱中正频率部分的峰值
positive_freqs = fft_frequencies[:n // 2]
positive_magnitude = np.abs(fft_values[:n // 2])
peak_frequency = positive_freqs[np.argmax(positive_magnitude)]
print(f"FFT初步频率估计：{peak_frequency} Hz")


# 步骤3：粒子群优化（PSO）算法
# 定义误差函数（适应度函数）
def error_function(f, t, r, A=1, phi=0):
    # A和phi未知，因此生成信号时进行归一化处理
    s = A * np.sin(2 * np.pi * f * t + phi)
    s_normalized = s / np.std(s)  # 对生成信号进行归一化
    r_normalized = r / np.std(r)  # 对接收信号进行归一化
    error = np.mean((r_normalized - s_normalized) ** 2)  # 计算均方误差
    return error


# 粒子群优化算法
# 参数：粒子数量，频率范围，最大迭代次数，接收信号，时间序列
def particle_swarm_optimization(num_particles, bounds, max_iter, r, t):
    positions = [random.uniform(bounds[0], bounds[1]) for _ in range(num_particles)]  # 初始频率
    velocities = [random.uniform(-1, 1) for _ in range(num_particles)]  # 初始速度
    pbest_positions = positions[:]  # 个体历史最优位置
    pbest_values = [error_function(f, t, r) for f in positions]  # 个体历史最优值

    gbest_position = pbest_positions[0]  # 全局最优位置
    gbest_value = pbest_values[0]  # 全局最优值

    # 迭代更新粒子位置和速度
    for iter in range(max_iter):
        inertia_weight = 0.9 - iter * (0.9 - 0.4) / max_iter  # 动态调整惯性权重
        cognitive_coefficient = 1.5  # 认知系数
        social_coefficient = 2.0  # 社会系数

        for i in range(num_particles):
            # 计算当前粒子的误差值
            current_value = error_function(positions[i], t, r)

            # 更新个体最优解
            if current_value < pbest_values[i]:
                pbest_positions[i] = positions[i]
                pbest_values[i] = current_value

            # 更新全局最优解
            if current_value < gbest_value:
                gbest_position = positions[i]
                gbest_value = current_value

        # 更新每个粒子的速度和位置
        for i in range(num_particles):
            velocity = (inertia_weight * velocities[i] +
                        cognitive_coefficient * random.random() * (pbest_positions[i] - positions[i]) +
                        social_coefficient * random.random() * (gbest_position - positions[i]))
            velocities[i] = velocity

            # 更新位置
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], bounds[0], bounds[1])  # 保证位置在搜索范围内

        print(f"第 {iter + 1} 次迭代: 全局最优频率 = {gbest_position}, 误差 = {gbest_value}")

    return gbest_position, gbest_value


# 步骤4：运行PSO
# 根据FFT分析结果设置PSO参数
num_particles = 200  # 粒子数量
bounds = (peak_frequency * 0.8, peak_frequency * 1.2)  # 在FFT估计频率附近进行搜索
max_iter = 500  # 最大迭代次数

# 使用滤波后的信号运行PSO
best_frequency, best_error = particle_swarm_optimization(num_particles, bounds, max_iter, received_signal_filtered,
                                                         time_filtered)

# 输出最佳频率和误差
print(f"使用PSO的最佳频率估计：{best_frequency} Hz, 误差: {best_error}")
