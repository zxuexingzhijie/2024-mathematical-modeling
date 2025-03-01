% 已知参数
A = 4;           % 信号幅度（这个值在FFT中不是必需的，但可以用于验证结果）
l = pi/4;           % 信号相位（同样，这个值在FFT中不是必需的）

% 从接收到的数据中提取采样间隔和信号值
dt = ReceivedSignalTime(2) - ReceivedSignalTime(1);

% 计算信号的FFT
N = length(ReceivedSignalValue); % 信号长度
Y = fft(ReceivedSignalValue);

% 计算双边频谱的频率轴
f = (0:N-1)*(1/N)/dt*10^(-3);

% 由于FFT结果是对称的，我们只关心前半部分（正频率部分）
f_positive = f(1:floor(N/2)+1);

% 计算双边频谱的幅度
P2 = abs(Y/N);
P1 = P2(1:floor(N/2)+1);
P1(2:end-1) = 2*P1(2:end-1);

% 找出最大幅度对应的频率（即信号的频率）
[~, idx_max] = max(P1);
estimated_f = f_positive(idx_max);

% 如果采样频率足够高，且信号频率不在奈奎斯特频率的一半附近，则估计结果是准确的
% 否则，可能需要考虑频率混叠或其他效应

% 显示估计的频率
fprintf('估计的频率为: %.6f Hz\n', estimated_f);

% 可视化结果
figure;
plot(f_positive, P1);
xlabel('频率 (Hz)');
ylabel('幅度');
title('信号的频谱');
grid on;
hold on;
% 在估计的频率处画一个垂直线
plot([estimated_f estimated_f], [0 max(P1)], 'r--', 'LineWidth', 2);
legend('频谱', '估计的频率');