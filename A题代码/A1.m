
format long;
filename = 'C:\Users\周岩珏\Desktop\pythonProject1\第一问.xlsx';  % Excel 文件路径
data = readtable(filename);  % 读取文件中的数据为表格格式


time = data.('Received Signal Time');  % 假设时间列名为 'Received Signal Time'
x_t = data.('Received Signal Value');  % 假设信号值列名为 'Received Signal Value'

% 显示提取的数据（可选）
disp(time(1:5));  % 显示前5个时间点
disp(x_t(1:5));   % 显示前5个信号值

%数据过大，仅取前5000条
A = 4; % 信号幅度
f = 630e3; % 信号频率（Hz）
l = pi/4; % 信号相位
dt = ReceivedSignalTime(2) - ReceivedSignalTime(1);
Fs = 1 / dt;
t = (0:length(ReceivedSignalValue)-1) * dt; % 时间向量
signal = A*sin(2*pi*f*t+l);
noise = ReceivedSignalValue' - signal;


% 计算噪声的统计特性
noise_mean = mean(noise);
noise_var = var(noise);
noise_std = std(noise);

% 输出噪声的统计特性
fprintf('噪声均值: %f\n', noise_mean);
fprintf('噪声方差: %f\n', noise_var);
fprintf('噪声标准差: %f\n', noise_std);

% 计算噪声的功率谱密度（PSD）
% 注意：pwelch 函数可能需要 Signal Processing Toolbox
[pxx, f_psd] = pwelch(noise, [], [], [], Fs);

% 绘制噪声的时间波形
figure;
plot(t, noise);
title('噪声的时间波形');
xlabel('时间 (s)');
ylabel('幅值');
% 绘制噪声的功率谱密度
figure;
plot(f_psd, 10*log10(pxx)); % 将PSD转换为dB单位进行绘制
title('噪声的功率谱密度');
xlabel('频率 (Hz)');
ylabel('功率/频率 (dB/Hz)');
xlim([0 Fs/2]); % 限制频率轴范围到奈奎斯特频率
%概率分布分析
% 假设noise是你的噪声数据向量
figure;
histogram(noise, 'Normalization', 'pdf'); % 绘制概率密度函数的直方图
xlabel('Noise Value');
ylabel('Probability Density');
title('Probability Distribution of Noise');

% 进行正态性检验（例如，Kolmogorov-Smirnov检验）
[h, p] = kstest(noise);
if h == 0
    disp('The noise data does not significantly differ from a normal distribution (p-value > 0.05).');
else
    disp('The noise data significantly differs from a normal distribution (p-value <= 0.05).');
end
%自相关函数分析
[acf, lags] = xcorr(noise, 'coeff'); % 计算自相关函数，并归一化
figure;
plot(lags, acf);
xlabel('Lags');
ylabel('Autocorrelation Coefficient');
title('Autocorrelation Function of Noise');

%频谱分析
N = length(noise); % 噪声数据的长度
Y = fft(noise); % 计算FFT
P2 = abs(Y/N); % 双边频谱的幅值
P1 = P2(1:N/2+1); % 单边频谱的幅值（对于实数信号）
P1(2:end-1) = 2*P1(2:end-1); % 由于FFT的对称性，只取一半并乘以2
f = Fs*(0:(N/2))/N*10^(-3); % 频率向量

figure;
plot(f, P1);
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Single-Sided Amplitude Spectrum of Noise');

%偏度与峰度分析
skewness_value = skewness(noise);
kurtosis_value = kurtosis(noise);

fprintf('Skewness: %.4f\n', skewness_value);
fprintf('Kurtosis: %.4f\n', kurtosis_value);