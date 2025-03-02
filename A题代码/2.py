import pandas as pd
import numpy as np
import random
from scipy.signal import butter, lfilter

# 步骤1：加载并过滤数据
# 加载Excel文件
file_path = r'C:\Users\***\Desktop\pythonProject1\第二题.xlsx'
data = pd.read_excel(file_path)

# 根据给定条件过滤数据
# 筛选条件：时间范围在0到0.0005之间，信号值在特定范围内
filtered_data = data[
    (data['Received Signal Time'] >= 0) &
    (data['Received Signal Time'] <= 0.0005) &
    (data['Received Signal Value'] >= -7.28567691506326) &
    (data['Received Signal Value'] <= 7.50775518925756)
]
# 提取过滤后的时间和信号值
time_filtered = filtered_data['Received Signal Time'].values
received_signal_filtered = filtered_data['Received Signal Value'].values

# 应用带通滤波器去除噪声
# 设计带通滤波器以去除信号中超出特定频段的噪声
# 滤波器设计函数 - Butterworth带通滤波器
# 参数：低截止频率，高截止频率，采样频率，滤波器阶数
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # 奈奎斯特频率，即采样频率的一半
    low = lowcut / nyquist  # 归一化低截止频率
    high = highcut / nyquist  # 归一化高截止频率
    b, a = butter(order, [low, high], btype='band')  # 使用Butterworth设计带通滤波器
    return b, a

# 带通滤波函数，用于对信号进行滤波
# 参数：数据，低截止频率，高截止频率，采样频率，滤波器阶数
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)  # 对信号进行滤波处理
    return y

# 设置带通滤波器参数
sampling_frequency = 2e8  # 假设采样频率为200 MHz
lowcut = 1e7  # 低截止频率为10 MHz
highcut = 5e7  # 高截止频率为50 MHz

# 对接收信号应用带通滤波器，去除不在频段内的频率成分
filtered_signal = bandpass_filter(received_signal_filtered, lowcut, highcut, sampling_frequency)

# 定义误差函数
# 计算估计信号与实际接收信号之间的误差
# 参数：频率，时间，接收信号，振幅，初相位
def error_function(f, t, r, A=2, phi=0):
    s = A * np.sin(2 * np.pi * f * t + phi)  # 使用给定频率生成正弦信号
    error = np.mean((r - s) ** 2)  # 计算信号与实际接收信号的均方误差
    return error

# 粒子群优化（PSO）算法
# 使用PSO寻找最优频率以最小化误差函数
# 参数：粒子数量，频率搜索范围，最大迭代次数，接收信号，时间序列，振幅，初相位
def particle_swarm_optimization(num_particles, bounds, max_iter, r, t, A=2, phi=0):
    # 初始化粒子的位置和速度
    positions = [random.uniform(bounds[0], bounds[1]) for _ in range(num_particles)]  # 初始频率随机分布在指定范围内
    velocities = [random.uniform(-1, 1) for _ in range(num_particles)]  # 初始速度，随机分布在-1到1之间
    pbest_positions = positions[:]  # 个体历史最优位置
    pbest_values = [error_function(f, t, r, A, phi) for f in positions]  # 计算每个粒子的个体最优值

    # 初始化全局最优解
    gbest_position = pbest_positions[0]
    gbest_value = pbest_values[0]

    # 迭代更新粒子的位置和速度
    for iter in range(max_iter):
        inertia_weight = 0.9 - iter * (0.9 - 0.4) / max_iter  # 动态调整惯性权重，从0.9逐渐减小到0.4

        # 动态调整认知和社会系数
        cognitive_coefficient = 1.5  # 认知系数，表示粒子向自身历史最优位置的吸引力
        social_coefficient = 2.0  # 社会系数，表示粒子向全局最优位置的吸引力

        for i in range(num_particles):
            # 计算当前粒子的误差值
            current_value = error_function(positions[i], t, r, A, phi)

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
            velocities[i] = velocity  # 更新粒子的速度

            # 更新粒子的位置
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], bounds[0], bounds[1])  # 保证粒子的位置在搜索范围内

        # 打印当前的全局最优解
        print(f"第 {iter + 1} 次迭代: 全局最优频率 = {gbest_position}, 误差 = {gbest_value}")

    return gbest_position, gbest_value

#  运行粒子群优化
# 设置PSO参数
num_particles = 200  # 增加粒子数量以提高搜索覆盖度
bounds = (1e7, 5e7)  # 缩小频率搜索范围以提高搜索效率
max_iter = 300  # 增加最大迭代次数以便充分搜索

# 使用滤波后的信号运行PSO寻找最优频率
best_frequency, best_error = particle_swarm_optimization(num_particles, bounds, max_iter, filtered_signal, time_filtered)

# 输出最佳频率和误差
print(f"最佳频率: {best_frequency}, 误差: {best_error}")

