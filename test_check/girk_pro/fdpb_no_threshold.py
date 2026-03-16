# girk_pro/fdpb_no_threshold.py
import numpy as np
from scipy.signal import get_window


def fdpb_no_threshold(
        x: np.ndarray,
        nfft: int = 1024,
        window='blackman',
        overlap_ratio: float = 0.5,
        dtype=np.complex64
):
    """
    自适应 FDPB 干扰抑制（无需门限）

    Parameters
    ----------
    x : np.ndarray[complex]
        输入复基带信号（1D）
    nfft : int
        FFT 点数
    window : str or np.ndarray
        窗函数
    overlap_ratio : float
        重叠比例 [0, 1)
    dtype : np.dtype
        输出数据类型

    Returns
    -------
    y : np.ndarray
        抑制干扰后的信号
    """
    if x.ndim != 1:
        raise ValueError("Input must be 1D.")
    if not (0 <= overlap_ratio < 1):
        raise ValueError("overlap_ratio must be in [0, 1).")

    N = len(x)
    if isinstance(window, str):
        w = get_window(window, nfft)
    else:
        w = np.asarray(window)
        if len(w) != nfft:
            raise ValueError("Window length must equal nfft.")

    hop = int(nfft * (1 - overlap_ratio))
    if hop < 1:
        hop = 1

    eps = 1e-12
    # 输出缓冲区
    y = np.zeros(N, dtype=dtype)
    gain = nfft / np.sum(w ** 2)  # 能量补偿

    # 帧数计算
    num_frames = max(1, (N + hop - 1) // hop)

    for m in range(num_frames):
        start = m * hop
        end = start + nfft

        # 提取帧（补零）
        if end <= N:
            frame = x[start:end].copy()
        else:
            frame = np.concatenate([x[start:], np.zeros(end - N, dtype=x.dtype)])

        # 加窗 + FFT
        X = np.fft.fft(frame * w)

        # 归一化
        X_mag = np.abs(X)
        X /= (X_mag + eps)

        # IFFT + 累加
        rec = np.fft.ifft(X)
        out_end = min(start + nfft, N)
        y[start:out_end] += (rec[:out_end - start] * gain).astype(dtype)

    return y