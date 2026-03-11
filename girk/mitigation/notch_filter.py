
# notch_filter.py
import numpy as np
from scipy.signal import get_window
from typing import Union, Tuple


def iir_notch(
        x: np.ndarray,
        fs: float,
        mu: float = 0.01,
        alpha: float = 0.9,
        return_freq_est: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Adaptive Notch Filter based on LMS Structure (Wendel et al., ION GNSS+ 2016).

    Fixed version with numerical stability enhancements.
    """
    if x.ndim != 1:
        raise ValueError("Input must be 1D.")
    if not np.iscomplexobj(x):
        raise ValueError("Input must be complex baseband signal.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")
    if mu <= 0:
        raise ValueError("mu must be positive.")

    N = len(x)
    dtype = x.dtype  # 保持输入 dtype
    y = np.zeros(N, dtype=dtype)
    f_est = np.zeros(N)

    # Initialize filter states with correct dtype
    r_prev = dtype.type(0.0 + 0.0j)
    z0 = dtype.type(0.0 + 0.0j)

    for n in range(N):
        xn = x[n]

        # --- IIR section: r[n] = x[n] + alpha * z0 * r[n-1]
        r = xn + alpha * z0 * r_prev

        # --- FIR section: y[n] = r[n] - z0 * r[n-1]
        y[n] = r - z0 * r_prev

        # --- Normalized LMS update (STABLE VERSION)
        # Compute gradient and update
        denom = np.abs(r) ** 2 + 1e-12
        delta_z0 = mu * y[n] * np.conj(r_prev) / denom
        z0 = z0 + delta_z0

        # Update state
        r_prev = r

        # --- STABILITY CONSTRAINT: |z0| <= 0.99 ---
        # z0_mag = np.abs(z0)
        # if z0_mag > 0.99:
        #     z0 = 0.99 * z0 / z0_mag  # Project onto circle of radius 0.99

        # --- Estimate frequency from z0
        angle = np.angle(z0)
        f_inst = (angle * fs) / (2 * np.pi)
        # Wrap to [-fs/2, fs/2]
        if f_inst > fs / 2:
            f_inst -= fs
        elif f_inst < -fs / 2:
            f_inst += fs
        f_est[n] = f_inst

    if return_freq_est:
        return y, f_est
    else:
        return y

def fir_notch(
        x: np.ndarray,
        nfft: int = 1024,
        threshold: float = 100.0,
        window: Union[str, tuple, np.ndarray] = 'blackman',
        overlap_ratio: float = 0.5,
        dtype: np.dtype = np.complex64
) -> np.ndarray:
    """
    频域脉冲消隐（Frequency Domain Pulse Blanking, FDPB）

    在频域检测超过门限的强干扰点，并将其置零，适用于抗窄带/脉冲干扰。

    Parameters
    ----------
    x : np.ndarray
        输入复基带信号（1D array，行或列向量均可）
    nfft : int, optional
        FFT 点数，默认 1024
    threshold : float, optional
        功率谱门限（线性值，非 dB），默认 100.0
    window : str, tuple, or np.ndarray, optional
        窗函数类型，支持：
        - 字符串：'blackman', 'hann', 'hamming', 'kaiser'
        - tuple：('kaiser', beta)
        - 自定义窗（长度必须为 nfft）
        默认 'blackman'
    overlap_ratio : float, optional
        重叠比例（0.0 ~ 1.0），默认 0.5（50% 重叠）
    dtype : np.dtype, optional
        输出数据类型，默认 np.complex64

    Returns
    -------
    y : np.ndarray
        输出信号，形状与输入 x 完全一致

    Examples
    --------
    >>> x = np.random.randn(12345) + 1j * np.random.randn(12345)
    >>> y = fdpb(x, nfft=512, threshold=50, window='hann')
    >>> assert y.shape == x.shape
    """
    # --- 参数校验 ---
    if x.ndim != 1:
        raise ValueError("Input x must be a 1D array (vector).")
    if nfft <= 0 or not isinstance(nfft, int):
        raise ValueError("nfft must be a positive integer.")
    if not (0.0 <= overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be in [0, 1).")
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")

    # 保存原始形状（用于恢复行/列向量）
    original_shape = x.shape
    x = x.ravel()  # 转为 1D 行向量（内部统一处理）
    N = len(x)

    # --- 构造窗函数 ---
    if isinstance(window, np.ndarray):
        if len(window) != nfft:
            raise ValueError(f"Custom window length ({len(window)}) must equal nfft ({nfft}).")
        w = window
    else:
        try:
            w = get_window(window, nfft)
        except Exception as e:
            raise ValueError(f"Invalid window specification: {window}") from e

    w = w.astype(np.float32)

    # --- 重叠参数 ---
    overlap = int(overlap_ratio * nfft)
    hop = nfft - overlap
    if hop <= 0:
        hop = 1  # 防止死循环

    # --- 预分配输出 ---
    y = np.zeros(N, dtype=dtype)

    # --- 能量归一化因子（补偿窗衰减）---
    window_power = np.sum(w ** 2)
    if window_power == 0:
        gain = 1.0
    else:
        gain = nfft / window_power  # 标准 OLA 归一化

    # --- 计算帧数（向上取整）---
    if N <= overlap:
        num_frames = 1
    else:
        num_frames = int(np.ceil((N - overlap) / hop))

    # --- 主循环：重叠-相加 (Overlap-Add) ---
    for m in range(num_frames):
        start = m * hop
        end = start + nfft

        # 提取当前帧（不足则补零）
        if end <= N:
            x_frame = x[start:end]
        else:
            x_frame = np.concatenate([x[start:], np.zeros(end - N, dtype=x.dtype)])

        # 加窗
        x_win = x_frame * w

        # FFT → 功率谱 → 脉冲消隐
        X = np.fft.fft(x_win)
        P = np.abs(X) ** 2  # 功率谱

        # 超过门限的频点置零
        mask = P >= threshold
        X[mask] = 0

        # IFFT 回时域
        x_rec = np.fft.ifft(X)

        # 仅累加有效部分（避免越界）
        out_start = start
        out_end = min(start + nfft, N)
        segment_len = out_end - out_start

        # 能量补偿 + 累加
        y[out_start:out_end] += (x_rec[:segment_len] * gain).astype(dtype)

    # 恢复原始形状（行/列向量）
    return y.reshape(original_shape)