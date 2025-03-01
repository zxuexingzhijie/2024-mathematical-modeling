% 已知数据点和插值点
x = ReceivedSignalTime(1:80); % 时间向量
y = ReceivedSignalValue(1:80); % 原始信号值
dt = x(2) - x(1);
Fs = 1 / dt;
% 绘制原始信号
figure;
%scatter(x, y);
plot(x,y);
xlabel('时间 (s)');
ylabel('信号值');
title('原始信号');

% 使用 linspace 创建插值点
xq = linspace(min(x), max(x),100); % 插值后的时间点

% 使用 interp1 函数进行样条插值
yq = interp1(x, y, xq, 'spline'); % 插值后的信号值

%yq=y;
% 绘制插值后的信号
figure;
plot(xq, yq);
xlabel('时间 (s)');
ylabel('信号值');
title('插值后的信号');

% 对插值后的信号进行 FFT 变换
Yq = fft(yq);

% FFT 结果的长度
N = length(Yq);

% 创建频率向量（只取到奈奎斯特频率）
f = (0:N-1)*(Fs/N)*10^(-3); % Fs 是采样频率，需要您提前定义或计算
f = f(1:N/2+1); % 由于 FFT 结果是对称的，通常只取前半部分

% 计算 FFT 幅度的绝对值，并取前半部分（因为 FFT 结果是对称的）
P = abs(Yq/N); % 归一化 FFT 幅度
P = P(1:N/2+1);
P(2:end-1) = 2*P(2:end-1); % 修正由于 FFT 对称性导致的幅度加倍（除了第一个和最后一个元素）

% 找到 FFT 结果中的峰值位置（索引）
[~, idx] = max(P);

% 计算对应的频率（峰值频率）
estimated_f = f(idx);

% 显示结果
fprintf('估计的频率为: %.2f Hz\n', estimated_f);

% 绘图（可选）
figure;
plot(f, P);
xlabel('频率 (Hz)');
ylabel('FFT 幅度');
title('插值后信号的 FFT');
%%
% 初始化存储每组预测频率的数组
predicted_frequencies = zeros(1, 10); % 因为有10组（800/80）
dt = x(2) - x(1);
Fs = 1 / dt;
% 循环处理每组信息
for group = 1:10
    % 提取当前组的信号值
    startIdx = (group-1)*80 + 1;
    endIdx = group*80;
    y = ReceivedSignalValue(startIdx:endIdx);
    
    % 执行FFT
    N = length(y); % 信号长度
    Y = fft(y);
    
    % 计算频率轴
    f = (0:N-1)*(Fs/N)/10^3;
    
    % 只取FFT结果的前半部分（对于实数输入信号）
    P = abs(Y/N); % 幅度谱
    f_half = f(1:N/2+1);
    P_half = P(1:N/2+1);
    
    % 找到峰值频率（这里简单地取最大值的索引对应的频率）
    [~, idx] = max(P_half);
    predicted_frequency = f_half(idx);
    
    % 存储预测频率
    predicted_frequencies(group) = predicted_frequency;
end

% 计算预测频率的平均值
average_predicted_frequency = mean(predicted_frequencies);

% 显示结果
disp(['平均预测频率: ', num2str(average_predicted_frequency), ' Hz']);