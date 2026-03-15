import numpy as np
from typing import Union
from scipy.signal import butter, filtfilt

# ===== 单音干扰生成 ======
def single_tone_interference(
    fs: float = 21e6,
    freq_offset: float = 0.0,
    duration: float = 0.01,
    power: float = 1.0,
    phase0: float = 0.0,
    dtype: Union[np.dtype, type] = np.complex64
) -> np.ndarray:
    """
    生成复基带下单音（纯正弦）干扰信号。

    这是一种理想化的窄带干扰，带宽趋近于零，
    数学模型为单一频率的复指数信号：

        j(t) = sqrt(power) * exp(j * (2π * freq_offset * t + phase0))

    注意：若需生成具有有限带宽的干扰（如滤波后的噪声），
    请使用其他模型，例如 'narrowband_noise_interference'。

    参数
    ----------
    fs : float, 可选
        采样频率，单位 Hz。默认为 21e6（即 21 Msps）。
    freq_offset : float, 可选
        相对于载波的频率偏移量，单位 Hz。
        正值表示干扰位于载波上方，负值表示下方。
        默认为 0.0 Hz（即位于基带中心）。
    duration : float, 可选
        信号持续时间，单位秒。默认为 0.01（即 10 毫秒）。
    power : float, 可选
        干扰功率（线性尺度，非 dB）。幅度为 sqrt(power)。默认为 1.0。
    phase0 : float, 可选
        初始相位，单位弧度。默认为 0.0。
    dtype : numpy.dtype, 可选
        输出数组的数据类型。默认为 np.complex64（节省内存）。

    返回
    -------
    np.ndarray
        复数值的单音干扰信号，形状为 (N,)，
        其中 N = int(fs * duration)。
    """
    # 输入参数校验
    if fs <= 0:
        raise ValueError("采样频率 'fs' 必须为正数。")
    if duration <= 0:
        raise ValueError("持续时间必须为正数。")
    if power < 0:
        raise ValueError("功率不能为负数。")

    # 计算样本点数
    n_samples = int(np.round(fs * duration))
    if n_samples == 0:
        raise ValueError("在给定采样率下，持续时间过短，无法生成有效信号。")

    # 生成时间向量
    t = np.arange(n_samples, dtype=np.float32) / fs

    # 生成单音复指数干扰信号
    angle = 2 * np.pi * freq_offset * t + phase0
    interference = np.sqrt(power) * np.exp(1j * angle)

    return interference.astype(dtype)

# ===== 窄带干扰生成 ======
def narrowband_interference(
    fs: float = 21e6,
    freq_offset: float = 0.0,
    bandwidth: float = 1e6,
    duration: float = 0.01,
    power: float = 1.0,
    phase0: float = 0.0,
    dtype: Union[np.dtype, type] = np.complex64
) -> np.ndarray:
    """
    生成具有有限带宽的窄带干扰信号（复基带），支持指定持续时间。

    实现流程：
        1. 预生成一段较长的复高斯白噪声（避免滤波瞬态）
        2. 用低通滤波器限制带宽（截止频率 = bandwidth / 2）
        3. 调制到指定频偏处
        4. 截取中间稳定段（长度 = duration）
        5. 归一化至目标功率

    参数
    ----------
    fs : float, 可选
        采样频率（Hz）。默认 21e6。
    freq_offset : float, 可选
        干扰中心频率相对于载波的偏移（Hz）。默认 0.0。
    bandwidth : float, 可选
        干扰信号的 **3 dB 带宽**（Hz），即功率谱密度下降 3 dB 时的总宽度。
        滤波器采用 4 阶 Butterworth 低通，中心频率为 0（基带），
        调制后干扰中心位于 `freq_offset`。
        干扰信号的等效噪声带宽（Hz）。必须满足 0 < bandwidth <= fs。
        默认 1e6（1 MHz）。
    duration : float, 可选
        干扰信号持续时间（秒）。默认 0.01（10 ms）。
    power : float, 可选
        干扰总功率（线性值，非 dB）。默认 1.0。
    phase0 : float, 可选
        初始相位（弧度）。默认 0.0。
    dtype : numpy.dtype, 可选
        输出数组的数据类型。默认 np.complex64。

    返回
    -------
    np.ndarray
        复数值窄带干扰信号，形状 (N,)，其中 N = int(fs * duration)。
    """
    # --- 参数校验 ---
    if fs <= 0:
        raise ValueError("采样频率 'fs' 必须为正数。")
    if duration <= 0:
        raise ValueError("持续时间 'duration' 必须为正数。")
    if bandwidth <= 0:
        raise ValueError("带宽 'bandwidth' 必须为正数。")
    if bandwidth > fs:
        raise ValueError("带宽不能超过采样率。")
    if power < 0:
        raise ValueError("功率不能为负数。")

    # --- 计算所需输出长度 ---
    n_output = int(np.round(fs * duration))
    if n_output == 0:
        raise ValueError("持续时间过短，无法生成有效信号。")

    # --- 确定内部生成长度（预留滤波过渡区）---
    # 过渡区长度 ≈ 4 * (fs / bandwidth) （经验规则，覆盖滤波器建立时间）
    transition_samples = int(4 * fs / bandwidth)
    n_internal = n_output + 2 * transition_samples
    n_internal = max(n_internal, 8192)  # 至少 8192 点

    # --- 1. 生成复高斯白噪声 ---
    noise_real = np.random.randn(n_internal)
    noise_imag = np.random.randn(n_internal)
    # 注意：此处不除 sqrt(2)，因为后续滤波会改变功率，最终统一归一化

    # --- 2. 设计低通滤波器（Butterworth，4阶）---
    nyquist = fs / 2
    cutoff_norm = min((bandwidth / 2) / nyquist, 0.999)  # 防止 >=1.0
    b, a = butter(N=4, Wn=cutoff_norm, btype='low', analog=False)

    # --- 3. 零相位滤波 ---
    filtered_real = filtfilt(b, a, noise_real)
    filtered_imag = filtfilt(b, a, noise_imag)
    baseband_narrow = filtered_real + 1j * filtered_imag

    # --- 4. 调制到目标频偏 ---
    t_internal = np.arange(n_internal) / fs
    carrier = np.exp(1j * (2 * np.pi * freq_offset * t_internal + phase0))
    interference_full = baseband_narrow * carrier

    # --- 5. 截取中间稳定段 ---
    start = transition_samples
    end = start + n_output
    interference = interference_full[start:end]

    # --- 6. 功率归一化（关键！在截取后进行）---
    current_power = np.mean(np.abs(interference)**2)
    if current_power == 0:
        raise RuntimeError("截取段功率为零，请增大带宽或时长。")
    scale = np.sqrt(power / current_power)
    interference *= scale

    return interference.astype(dtype)


# ===== 线性扫频干扰生成 ======
def linear_chirp_interference(
        fs: float = 21e6,
        freq_offset: float = 0.0,
        sweep_bandwidth: float = 5e6,
        duration: float = 0.01,
        sweep_period: float = 1e-3,
        power: float = 1.0,
        phase0: float = 0.0,
        dtype: Union[np.dtype, type] = np.complex64
) -> np.ndarray:
    """
    生成复基带下的线性扫频干扰信号（Sawtooth Chirp / Linear FM）。

    该信号的特征是瞬时频率随时间呈周期性线性变化（锯齿波形）：
    在每个扫频周期内，频率从起始频率 (f_start) 线性增加到终止频率 (f_end)，
    然后瞬间跳回 f_start 开始下一个周期。

    频率范围定义：
        f_start = freq_offset - sweep_bandwidth / 2
        f_end   = freq_offset + sweep_bandwidth / 2

    数学模型：
        j(t) = sqrt(power) * exp(j * (2π * ∫f(τ)dτ + phase0))

    其中 f(t) 是周期为 sweep_period 的锯齿波频率函数。

    参数
    ----------
    fs : float, 可选
        采样频率，单位 Hz。默认为 21e6。
    freq_offset : float, 可选
        扫频中心的频率偏移量，单位 Hz。默认为 0.0。
        扫频范围将以该频率为中心对称分布。
    sweep_bandwidth : float, 可选
        扫频总带宽，单位 Hz。即 (f_end - f_start)。
        必须为正数且小于采样率。默认为 5e6 (5 MHz)。
    duration : float, 可选
        信号持续时间，单位秒。默认为 0.01。
    sweep_period : float, 可选
        扫频信号的周期，单位秒。即完成一次从 f_start 到 f_end 扫描所需的时间。
        必须大于 0 且小于 duration。默认为 1e-3 (1 ms)。
    power : float, 可选
        干扰功率（线性尺度）。默认为 1.0。
    phase0 : float, 可选
        初始相位，单位弧度。默认为 0.0。
    dtype : numpy.dtype, 可选
        输出数组的数据类型。默认为 np.complex64。

    返回
    -------
    np.ndarray
        复数值的线性扫频干扰信号，形状为 (N,)，
        其中 N = int(fs * duration)。

     Raises
    ------
    ValueError
        当参数不满足物理约束时抛出。
    """
    # --- 参数校验 ---
    if fs <= 0:
        raise ValueError("采样频率 'fs' 必须为正数。")
    if duration <= 0:
        raise ValueError("持续时间 'duration' 必须为正数。")
    if sweep_bandwidth <= 0:
        raise ValueError("扫频带宽 'sweep_bandwidth' 必须为正数。")
    if sweep_bandwidth >= fs:
        raise ValueError("扫频带宽不能超过采样率（需满足奈奎斯特采样定理）。")
    if sweep_period <= 0:
        raise ValueError("扫频周期 'sweep_period' 必须为正数。")
    if sweep_period > duration:
        raise ValueError("扫频周期不能大于信号总持续时间。")
    if power < 0:
        raise ValueError("功率不能为负数。")

    # --- 计算样本点数 ---
    n_samples = int(np.round(fs * duration))
    if n_samples == 0:
        raise ValueError("在给定采样率下，持续时间过短，无法生成有效信号。")

    # --- 生成时间向量 ---
    t = np.arange(n_samples, dtype=np.float32) / fs

    # --- 计算扫频边界 ---
    f_start = freq_offset - sweep_bandwidth / 2.0
    f_end = freq_offset + sweep_bandwidth / 2.0

    # --- 计算瞬时相位 ---
    # 方法：利用 sawtooth (锯齿波) 生成归一化的时间进度 (0 到 1)，
    # 然后映射到频率变化，最后积分得到相位。
    #
    # 对于线性扫频：f(t) = f_start + k * (t % T)
    # 其中斜率 k = (f_end - f_start) / T = sweep_bandwidth / sweep_period
    # 相位 φ(t) = 2π * ∫f(t)dt = 2π * [f_start * t + 0.5 * k * (t % T)^2]

    # 1. 计算当前时间在周期内的位置 (t_mod = t % sweep_period)
    t_mod = np.remainder(t, sweep_period)

    # 2. 计算调频斜率 (Chirp Rate)
    chirp_rate = sweep_bandwidth / sweep_period

    # 3. 计算相位项
    # 第一项：中心频率或起始频率带来的线性相位增长
    # 第二项：扫频带来的二次相位增长 (0.5 * rate * t^2)
    # 注意：这里我们直接使用积分公式，确保相位连续
    # φ(t) = 2π * (f_start * t + 0.5 * chirp_rate * t_mod^2) + phase0

    phase = 2 * np.pi * (f_start * t + 0.5 * chirp_rate * (t_mod ** 2)) + phase0

    # --- 生成信号 ---
    interference = np.sqrt(power) * np.exp(1j * phase)

    return interference.astype(dtype)


# generators.py（开发时临时测试用）
# if __name__ == "__main__":
#     # 测试新生成的扫频干扰
#     import matplotlib.pyplot as plt
#
#     # 生成一个短时的扫频信号用于验证
#     fs_test = 21e6
#     dur_test = 2e-3  # 2ms
#     bw_test = 10e6  # 10MHz
#     period_test = 0.5e-3  # 0.5ms
#
#     sig = linear_chirp_interference(
#         fs=fs_test,
#         freq_offset=2e6,
#         sweep_bandwidth=bw_test,
#         duration=dur_test,
#         sweep_period=period_test,
#         power=1.0
#     )
#
#     print(f"已生成 {len(sig)} 个扫频干扰样本点")
#     print(f"理论周期数：{dur_test / period_test}")
#
#     # 简单的频谱检查 (可选，需安装 matplotlib)
#     try:
#         from scipy.signal import spectrogram
#
#         f, t_spec, Sxx = spectrogram(sig, fs=fs_test, nperseg=256)
#         plt.figure(figsize=(10, 4))
#         plt.pcolormesh(t_spec, f / 1e6, 10 * np.log10(Sxx + 1e-12), shading='gouraud')
#         plt.ylabel('Frequency [MHz]')
#         plt.xlabel('Time [sec]')
#         plt.title('Linear Chirp Interference Spectrogram')
#         plt.ylim(-15, 15)  # 显示中心附近
#         plt.colorbar(label='Power/Frequency [dB/Hz]')
#         plt.grid(axis='y')
#         plt.show()
#     except ImportError:
#         print("未安装 matplotlib 或 scipy，跳过频谱绘图。")

# generators.py（开发时临时测试用）
if __name__ == "__main__":
    interference = narrowband_interference(fs=20e6, freq_offset=1e6, duration=0.01)
    print(f"已生成 {len(interference)} 个样本点")