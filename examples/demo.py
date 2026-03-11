"""
# FDPB 抗干扰性能演示

展示 **频域脉冲消隐（FDPB）** 算法对以下两类干扰的抑制效果：
- 单音干扰（Single-Tone Interference）
- 窄带干扰（Narrowband Interference）
使用 `girk` 工具包生成干扰信号，并应用 `fdpb` 进行抑制。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram

# 导入 girk 模块
from girk.interference.generators import single_tone_interference, narrowband_interference
from girk.mitigation import fdpb, iir_notch

# 设置全局绘图风格
plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (12, 4),
    'axes.grid': True
})

# 通用参数
fs = 21e6  # 采样率 (21 MHz)
duration = 0.01  # 信号时长 (10 ms)
noise_power = 1.0  # 热噪声功率

# 干扰参数
freq_offset = 1.5e6  # 干扰中心频偏 (1.5 MHz)
inr_db = 30.0            # 干噪比 (Interference-to-Noise Ratio, in dB)

inr_linear = 10**(inr_db / 10)   # 转换为线性值
interference_power = noise_power * inr_linear  # 干扰功率

# FDPB 参数
nfft = 1024
threshold = 100.0  # 门限：略高于噪声功率谱（实测调整）

# 1. 单音干扰场景
# 生成热噪声
np.random.seed(42)  # 仅用于可复现性（实际仿真应移除）
t = np.arange(int(fs * duration)) / fs
noise = (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * np.sqrt(noise_power / 2)

# 生成单音干扰
tone = single_tone_interference(
    fs=fs, freq_offset=freq_offset, duration=duration,
    power=interference_power, dtype=np.complex64
)

# 合成接收信号
x_tone = noise + tone

# 应用 FDPB
y_tone = fdpb(x_tone, nfft=nfft, threshold=threshold, window='blackman')
print(f"Single-tone interference scenario: input length = {len(x_tone)}, output length = {len(y_tone)}")


#  1.1 时域波形对比（实部）
plt.figure(figsize=(14, 3))
samples_show = 2000  # 显示前 2000 点
plt.plot(t[:samples_show] * 1e3, np.real(x_tone[:samples_show]), label='Original signal (with interference)', alpha=0.8)
plt.plot(t[:samples_show] * 1e3, np.real(y_tone[:samples_show]), label='FDPB output', alpha=0.9)
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (Real part)')
plt.title('Time-Domain Waveform Before and After Interference Suppression')
plt.legend()
plt.tight_layout()
plt.show()


# # 1.2 功率谱密度（PSD）对比
def plot_psd(signal, fs, label, color='C0'):
    f, psd = welch(signal, fs=fs, nperseg=8192, return_onesided=False)
    # 转换为 dB-Hz（相对于 1 Hz 带宽）
    psd_db = 10 * np.log10(psd)
    # 频率搬移到 [-fs/2, fs/2]
    f_shifted = np.fft.fftshift(f)
    psd_db_shifted = np.fft.fftshift(psd_db)
    plt.plot(f_shifted / 1e6, psd_db_shifted, color=color, label=label)

plt.figure(figsize=(14, 4))
plot_psd(x_tone, fs, 'Original signal', 'red')
plot_psd(y_tone, fs, 'FDPB output', 'blue')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power Spectral Density (dB/Hz)')
plt.title('PSD Comparison Before and After Interference Suppression')
plt.legend()
plt.xlim(-5, 5)
plt.tight_layout()
plt.show()

#  1.3 语谱图（Spectrogram）对比
def plot_spectrogram(signal, title, fs):
    f, t_spec, Sxx = spectrogram(
        signal, fs=fs, window='hann', nperseg=512,
        noverlap=256, return_onesided=False, scaling='density'
    )
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    plt.pcolormesh(t_spec * 1e3, f / 1e6, 10 * np.log10(Sxx),
                   shading='gouraud', cmap='viridis')
    plt.colorbar(label='PSD (dB/Hz)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (MHz)')
    plt.title(title)
    plt.ylim(-3, 3)

plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plot_spectrogram(x_tone, 'Original signal', fs)
plt.subplot(1, 2, 2)
plot_spectrogram(y_tone, 'FDPB output', fs)
plt.tight_layout()
plt.show()


## 2. 窄带干扰场景

# 生成窄带干扰
narrow = narrowband_interference(
    fs=fs, freq_offset=freq_offset, bandwidth=100e3,  # 100 kHz 带宽
    duration=duration, power=interference_power, dtype=np.complex64
)

# 合成接收信号
x_narrow = noise + narrow

y_narrow = fdpb(x_narrow, nfft=nfft, threshold=threshold, window='blackman')
print(f"Narrowband interference scenario: input length = {len(x_narrow)}, output length = {len(y_narrow)}")

#  2.1 时域波形对比（实部）
plt.figure(figsize=(14, 3))
samples_show = 2000  # 显示前 2000 点
plt.plot(t[:samples_show] * 1e3, np.real(x_narrow[:samples_show]), label='Original signal (with interference)', alpha=0.8)
plt.plot(t[:samples_show] * 1e3, np.real(y_narrow[:samples_show]), label='FDPB output', alpha=0.9)
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (Real part)')
plt.title('Time-Domain Waveform Before and After Interference Suppression')
plt.legend()
plt.tight_layout()
plt.show()

# # 2.2 功率谱密度（PSD）对比
def plot_psd(signal, fs, label, color='C0'):
    f, psd = welch(signal, fs=fs, nperseg=8192, return_onesided=False)
    # 转换为 dB-Hz（相对于 1 Hz 带宽）
    psd_db = 10 * np.log10(psd)
    # 频率搬移到 [-fs/2, fs/2]
    f_shifted = np.fft.fftshift(f)
    psd_db_shifted = np.fft.fftshift(psd_db)
    plt.plot(f_shifted / 1e6, psd_db_shifted, color=color, label=label)

plt.figure(figsize=(14, 4))
plot_psd(x_narrow, fs, 'Original signal', 'red')
plot_psd(y_narrow, fs, 'FDPB output', 'blue')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power Spectral Density (dB/Hz)')
plt.title('PSD Comparison Before and After Interference Suppression')
plt.legend()
plt.xlim(-5, 5)
plt.tight_layout()
plt.show()

#  3.3 语谱图（Spectrogram）对比
def plot_spectrogram(signal, title, fs):
    f, t_spec, Sxx = spectrogram(
        signal, fs=fs, window='hann', nperseg=512,
        noverlap=256, return_onesided=False, scaling='density'
    )
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    plt.pcolormesh(t_spec * 1e3, f / 1e6, 10 * np.log10(Sxx),
                   shading='gouraud', cmap='viridis')
    plt.colorbar(label='PSD (dB/Hz)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (MHz)')
    plt.title(title)
    plt.ylim(-3, 3)

plt.figure(figsize=(14, 4))
plt.subplot(1, 2, 1)
plot_spectrogram(x_narrow, 'Original signal', fs)
plt.subplot(1, 2, 2)
plot_spectrogram(y_narrow, 'FDPB output', fs)
plt.tight_layout()
plt.show()