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


# generators.py（开发时临时测试用）
if __name__ == "__main__":
    interference = narrowband_interference(fs=20e6, freq_offset=1e6, duration=0.01)
    print(f"已生成 {len(interference)} 个样本点")