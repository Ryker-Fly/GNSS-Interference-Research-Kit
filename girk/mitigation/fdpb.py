# fdpb.py
import numpy as np
from scipy.signal import get_window
from typing import Union, Optional


def fdpb(
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