
import matplotlib
# 强制使用 TkAgg 后端，这是 PyCharm 最通用的交互模式
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from matplotlib.animation import FuncAnimation

# 导入 girk 模块
from girk.interference.generators import linear_chirp_interference
from girk.mitigation import iir_notch

# =======  参数设置  =========
# 通用参数
fs = 21e6  # 采样率 (21 MHz)
duration = 0.05  # 信号时长 (10 ms)
noise_power = 1.0  # 热噪声功率

# 干扰参数
freq_offset = 1.5e6  # 干扰中心频偏 (1.5 MHz)
sweep_bandwidth = 12e6   # 扫频带宽
sweep_period = 20e-3      # 扫频周期
inr_db = 50.0            # 干噪比 (Interference-to-Noise Ratio, in dB)

inr_linear = 10**(inr_db / 10)   # 转换为线性值
interference_power = noise_power * inr_linear  # 干扰功率

# 生成热噪声
# np.random.seed(42)  # 仅用于可复现性（实际仿真应移除）
t = np.arange(int(fs * duration)) / fs
noise = (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * np.sqrt(noise_power / 2)

# 生成扫频干扰
interference = linear_chirp_interference(
    fs=fs,
    freq_offset=freq_offset,
    sweep_bandwidth=sweep_bandwidth,
    duration=duration,
    sweep_period=sweep_period,
    power=interference_power,
    phase0=0.0
)

# 合成接收信号
x_tone = noise + interference

# Apply IIR adaptive notch filter
# Note: iir_notch requires convergence time (~1000-5000 samples)
# iir_notch 参数
mu = 0.05        # learning rate
alpha = 0.85      # Narrower notch
y_tone, f_est_tone = iir_notch(
    x_tone,
    fs=fs,
    mu=mu,        # Adjusted learning rate
    alpha=alpha,      # Narrower notch
    return_freq_est=True
)
# 计算理论瞬时频率
k = sweep_bandwidth / sweep_period
f_true = freq_offset - (sweep_bandwidth / 2) + k * (t % sweep_period)

# 设置收敛偏移量（丢弃前 10ms 或 5000 点，视情况而定）
conv_offset = 10000

# 定义绘图用的切片数据
t_plot = t[conv_offset:]
x_plot = x_tone[conv_offset:]
y_plot = y_tone[conv_offset:]
f_est_plot = f_est_tone[conv_offset:]
f_true_plot = f_true[conv_offset:]  # 修正：这里定义 f_true_plot

# ======= 2. 动态绘图设置 (双子图) =======
fig, (ax_psd, ax_time) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.4)

nperseg = 1024
f_axis = np.fft.fftshift(np.fft.fftfreq(nperseg, 1 / fs))

# --- 子图1: 功率谱 ---
line_psd_raw, = ax_psd.plot([], [], color='red', alpha=0.5, label='Original signal (with interference)')
line_psd_filt, = ax_psd.plot([], [], color='blue', linewidth=1.2, label='IIR Notch Output')
ax_psd.set_xlim(-fs / 2, fs / 2)
ax_psd.set_ylim(-90, 40)
ax_psd.set_ylabel('PSD (dB/Hz)')
ax_psd.set_xlabel('Frequency (Hz)')
ax_psd.set_title('Frequency Domain: Power Spectral Density')
ax_psd.legend(loc='upper right')
ax_psd.grid(True, linestyle=':')

# --- 子图2: 时域波形 (显示实部) ---
line_time_raw, = ax_time.plot([], [], color='red', alpha=0.3, label='Original signal (with interference)')
line_time_filt, = ax_time.plot([], [], color='blue', alpha=0.8, label='IIR Notch Output')
ax_time.set_ylim(-np.max(np.abs(x_plot.real)) * 1.1, np.max(np.abs(x_plot.real)) * 1.1)
ax_time.set_ylabel('Amplitude (Real)')
ax_time.set_xlabel('Time (ms)')
ax_time.set_title('Time Domain: Signal Envelope')
ax_time.legend(loc='upper right')
ax_time.grid(True, linestyle=':')

# 动画控制参数
frame_step = 4000  # 增加步进以提高流畅度
window_size = 2048  # 计算频谱的窗口
time_window = 4000  # 时域显示的窗口长度


def update(frame):
    end_idx = frame * frame_step
    if end_idx < time_window: return line_psd_raw, line_psd_filt, line_time_raw, line_time_filt

    start_idx = end_idx - time_window

    # 1. 更新功率谱 (取最近的 window_size 个点)
    _, psd_raw = welch(x_plot[end_idx - window_size:end_idx], fs, nperseg=nperseg, return_onesided=False)
    _, psd_filt = welch(y_plot[end_idx - window_size:end_idx], fs, nperseg=nperseg, return_onesided=False)

    psd_raw_db = 10 * np.log10(np.fft.fftshift(psd_raw) + 1e-12)
    psd_filt_db = 10 * np.log10(np.fft.fftshift(psd_filt) + 1e-12)

    line_psd_raw.set_data(f_axis, psd_raw_db)
    line_psd_filt.set_data(f_axis, psd_filt_db)

    # 2. 更新时域图
    curr_t = t_plot[start_idx:end_idx] * 1000  # 换算成ms
    line_time_raw.set_data(curr_t, x_plot[start_idx:end_idx].real)
    line_time_filt.set_data(curr_t, y_plot[start_idx:end_idx].real)
    ax_time.set_xlim(curr_t[0], curr_t[-1])

    return line_psd_raw, line_psd_filt, line_time_raw, line_time_filt


total_frames = len(t_plot) // frame_step
ani = FuncAnimation(fig, update, frames=total_frames, interval=30, blit=True)

# ======= 动画保存部分 =======
# 建议先调大 frame_step (如 20000)，否则生成的视频文件会巨大
print("正在保存动画，请稍候...")

# 方法 A：保存为 MP4 (推荐，画质好，体积小)
# 需要系统中安装了 ffmpeg
try:
    ani.save('interference_mitigation.mp4', writer='ffmpeg', fps=20, dpi=100)
    print("MP4 保存成功！")
except Exception as e:
    print(f"MP4 保存失败: {e}")

# 方法 B：保存为 GIF (知乎直接拖入即可，但注意文件大小)
# 如果 MP4 不行，尝试这个
try:
    ani.save('interference_mitigation.gif', writer='pillow', fps=20)
    print("GIF 保存成功！")
except Exception as e:
    print(f"GIF 保存失败: {e}")

plt.show()