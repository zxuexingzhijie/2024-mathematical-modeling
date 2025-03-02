[中文版本](README_zh.md)

# 2024-mathematical-modeling

2024 "Shuwei Cup" Problem A: Frequency Estimation Problem in Aircraft Laser Speed Measurement

## Problem Solving Ideas:

#### Question 1:

* Read the signal data from the file and construct a known signal model. By subtracting the model signal from the original signal, the noise part is extracted.
* Then, perform statistical analysis on the noise by **calculating its mean, variance, and autocorrelation function** to understand the noise's intensity and temporal correlation.
* Next, perform frequency-domain analysis by calculating the noise's power spectral density through **Fourier transform**, revealing its frequency characteristics. Finally, use charts to display these results, including the **signal and model comparison chart, noise time-domain waveform, autocorrelation function, and power spectral density chart**, for a comprehensive analysis of the signal and noise characteristics.

![image-20250302115638403](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503021156476.png)

![image-20250302115658056](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503021156106.png)

#### Question 2:

* Read the signal data and filter out the portions that fall within the specified time and amplitude range. Then, apply a **Butterworth bandpass filte**r to the signal to remove noise outside the target frequency band.
* Next, **define an error function** to evaluate the fit between the sine wave generated at a given frequency and the filtered signal.
* Then, use the **Particle Swarm Optimization (PSO) algorithm** to search for the optimal frequency within the predefined frequency range, minimizing the error function. PSO **iteratively** adjusts the positions and velocities of particles, combining individual and global best solutions to continuously optimize the search results. Finally, output the **best matching frequency** and its corresponding **error value**, thereby determining the **optimal feature parameters** of the signal.

#### Question 3:

- Load the signal data from flight phase 3 and perform **Fast Fourier Transform** (FFT) for frequency spectrum analysis to initially estimate the dominant frequency of the signal.
- Define an **error function** to calculate the mean squared error between the sine wave generated at a given frequency and the actual received signal.
- Use the **Particle Swarm Optimization** (PSO) algorithm to search within the frequency range estimated by the FFT to minimize the error function and find the optimal frequency. The program outputs the **best frequency and its corresponding error value** obtained through PSO optimization, providing a more accurate frequency estimate.

![image-20250302121026724](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503021210801.png)

### Question 4:

- Load the signal data from flight phase 4 and perform **data cleaning** by removing invalid data points.
- Use **linear interpolation** to reconstruct the signal and **fill in the intermittent missing** parts.
- Perform **Fast Fourier Transform** (FFT) on the reconstructed signal for frequency spectrum analysis, initially estimating the dominant frequency of the signal. Then, define an error function and use the **Particle Swarm Optimization** (PSO) algorithm to **optimize the frequency estimation within the frequency range initially estimated by the FFT**. Output the best frequency and its corresponding error value obtained through PSO optimization, providing an accurate frequency estimate.

![image-20250302121346002](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503021213076.png)

![image-20250302121402681](https://typora-oss-picgo.oss-cn-beijing.aliyuncs.com/202503021214739.png)

#### Notes:

The paper is not convenient to be displayed here, but feel free to contact me privately if needed.

- **QQ: 2762006003 (for communication only, strictly confidential, no charges)**
