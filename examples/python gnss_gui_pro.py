import sys
import os
import numpy as np

# ===== Qt 兼容导入：优先 PySide6，其次 PyQt5 =====
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QComboBox,
        QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
        QMessageBox, QCheckBox, QSpinBox, QFileDialog, QSplitter
    )
    from PySide6.QtCore import QTimer, Qt
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QComboBox,
        QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
        QMessageBox, QCheckBox, QSpinBox, QFileDialog, QSplitter
    )
    from PyQt5.QtCore import QTimer, Qt

# ===== Matplotlib =====
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ===== scipy =====
from scipy.signal import spectrogram
from scipy.io import savemat

# ===== 你的干扰生成器 =====
from girk.interference.generators import (
    single_tone_interference,
    narrowband_interference,
    linear_chirp_interference
)


# =========================================================
# 工具函数
# =========================================================

def safe_eval_float(text: str, name: str) -> float:
    try:
        return float(eval(text, {"__builtins__": {}}, {}))
    except Exception:
        raise ValueError(f"{name} 输入无效: {text}")


def safe_eval_int(text: str, name: str) -> int:
    try:
        return int(float(eval(text, {"__builtins__": {}}, {})))
    except Exception:
        raise ValueError(f"{name} 输入无效: {text}")


def complex_awgn(n_samples: int, noise_power: float, dtype=np.complex64) -> np.ndarray:
    """
    生成复高斯白噪声，使得 E[|n|^2] = noise_power
    """
    if n_samples <= 0:
        raise ValueError("n_samples 必须为正整数")
    if noise_power < 0:
        raise ValueError("noise_power 不能为负数")

    if noise_power == 0:
        return np.zeros(n_samples, dtype=dtype)

    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    return noise.astype(dtype)


def compute_spectrum(x: np.ndarray, fs: float):
    """
    计算 FFT 幅度谱（dB）
    """
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])

    window = np.hanning(n)
    xw = x * window
    X = np.fft.fftshift(np.fft.fft(xw))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    mag_db = 20 * np.log10(np.abs(X) + 1e-12)
    return f, mag_db


def compute_spectrogram_iq(x: np.ndarray, fs: float, nperseg: int = 256, noverlap: int = 192):
    """
    对复基带信号计算时频图
    """
    if len(x) < 16:
        return np.array([]), np.array([]), np.array([[]])

    nperseg = min(nperseg, len(x))
    noverlap = min(noverlap, max(0, nperseg - 1))

    f, t, Sxx = spectrogram(
        x,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        return_onesided=False,
        scaling='density',
        mode='psd'
    )

    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    return f, t, Sxx_db


# =========================================================
# Matplotlib 画布
# =========================================================

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax.grid(True, alpha=0.3)


# =========================================================
# 主窗口
# =========================================================

class GNSSInterferenceProGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("GNSS 干扰信号生成与可视化 - 升级版")
        self.resize(1450, 860)

        self.signal = None
        self.interference = None
        self.noise = None
        self.fs = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.generate_and_plot)

        self._build_ui()
        self._update_param_visibility()

    # -------------------------
    # UI 构建
    # -------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout = QHBoxLayout(central)
        main_layout.addWidget(splitter)

        # ===== 左侧参数面板 =====
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # --- 信号参数 ---
        signal_group = QGroupBox("信号参数")
        signal_form = QFormLayout()

        self.type_combo = QComboBox()
        self.type_combo.addItems(["单音干扰", "窄带干扰", "线性扫频干扰"])
        self.type_combo.currentIndexChanged.connect(self._update_param_visibility)
        signal_form.addRow("干扰类型", self.type_combo)

        self.fs_edit = QLineEdit("21e6")
        signal_form.addRow("采样率 fs (Hz)", self.fs_edit)

        self.duration_edit = QLineEdit("0.002")
        signal_form.addRow("持续时间 (s)", self.duration_edit)

        self.freq_offset_edit = QLineEdit("0.0")
        signal_form.addRow("频偏 (Hz)", self.freq_offset_edit)

        self.power_edit = QLineEdit("1.0")
        signal_form.addRow("干扰功率", self.power_edit)

        self.phase0_edit = QLineEdit("0.0")
        signal_form.addRow("初始相位 (rad)", self.phase0_edit)

        self.bandwidth_label = QLabel("带宽 (Hz)")
        self.bandwidth_edit = QLineEdit("1e6")
        signal_form.addRow(self.bandwidth_label, self.bandwidth_edit)

        self.sweep_bw_label = QLabel("扫频带宽 (Hz)")
        self.sweep_bw_edit = QLineEdit("5e6")
        signal_form.addRow(self.sweep_bw_label, self.sweep_bw_edit)

        self.sweep_period_label = QLabel("扫频周期 (s)")
        self.sweep_period_edit = QLineEdit("1e-3")
        signal_form.addRow(self.sweep_period_label, self.sweep_period_edit)

        signal_group.setLayout(signal_form)
        left_layout.addWidget(signal_group)

        # --- 噪声与显示参数 ---
        display_group = QGroupBox("显示与噪声")
        display_form = QFormLayout()

        self.noise_power_edit = QLineEdit("0.05")
        display_form.addRow("热噪声功率", self.noise_power_edit)

        self.display_combo = QComboBox()
        self.display_combo.addItems(["时域波形", "频谱", "时频图"])
        display_form.addRow("显示模式", self.display_combo)

        self.view_samples_edit = QLineEdit("4096")
        display_form.addRow("显示样本点数", self.view_samples_edit)

        self.nperseg_edit = QLineEdit("256")
        display_form.addRow("时频窗长", self.nperseg_edit)

        self.noverlap_edit = QLineEdit("192")
        display_form.addRow("时频重叠", self.noverlap_edit)

        display_group.setLayout(display_form)
        left_layout.addWidget(display_group)

        # --- 动态刷新 ---
        refresh_group = QGroupBox("动态刷新")
        refresh_form = QFormLayout()

        self.auto_refresh_check = QCheckBox("启用自动刷新")
        self.auto_refresh_check.stateChanged.connect(self._toggle_auto_refresh)
        refresh_form.addRow(self.auto_refresh_check)

        self.refresh_ms_spin = QSpinBox()
        self.refresh_ms_spin.setRange(50, 10000)
        self.refresh_ms_spin.setValue(500)
        self.refresh_ms_spin.setSuffix(" ms")
        refresh_form.addRow("刷新周期", self.refresh_ms_spin)

        refresh_group.setLayout(refresh_form)
        left_layout.addWidget(refresh_group)

        # --- 操作按钮 ---
        btn_group = QGroupBox("操作")
        btn_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        self.generate_btn = QPushButton("生成并显示")
        self.generate_btn.clicked.connect(self.generate_and_plot)
        row1.addWidget(self.generate_btn)

        self.clear_btn = QPushButton("清空图像")
        self.clear_btn.clicked.connect(self.clear_plot)
        row1.addWidget(self.clear_btn)

        btn_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.save_npy_btn = QPushButton("保存 .npy")
        self.save_npy_btn.clicked.connect(lambda: self.save_signal("npy"))
        row2.addWidget(self.save_npy_btn)

        self.save_bin_btn = QPushButton("保存 .bin")
        self.save_bin_btn.clicked.connect(lambda: self.save_signal("bin"))
        row2.addWidget(self.save_bin_btn)

        btn_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.save_mat_btn = QPushButton("保存 .mat")
        self.save_mat_btn.clicked.connect(lambda: self.save_signal("mat"))
        row3.addWidget(self.save_mat_btn)

        self.stop_btn = QPushButton("停止自动刷新")
        self.stop_btn.clicked.connect(self.stop_auto_refresh)
        row3.addWidget(self.stop_btn)

        btn_layout.addLayout(row3)

        btn_group.setLayout(btn_layout)
        left_layout.addWidget(btn_group)

        # --- 状态信息 ---
        status_group = QGroupBox("状态")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("状态：待生成")
        self.status_label.setWordWrap(True)
        self.status_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)
        left_layout.addWidget(status_group)

        left_layout.addStretch()

        # ===== 右侧绘图区 =====
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        plot_group = QGroupBox("图形显示")
        plot_layout = QVBoxLayout()

        self.canvas = MplCanvas(self)
        plot_layout.addWidget(self.canvas)

        plot_group.setLayout(plot_layout)
        right_layout.addWidget(plot_group)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([380, 980])

    # -------------------------
    # 参数显隐
    # -------------------------
    def _update_param_visibility(self):
        mode = self.type_combo.currentText()

        is_nb = (mode == "窄带干扰")
        is_chirp = (mode == "线性扫频干扰")

        self.bandwidth_label.setVisible(is_nb)
        self.bandwidth_edit.setVisible(is_nb)

        self.sweep_bw_label.setVisible(is_chirp)
        self.sweep_bw_edit.setVisible(is_chirp)

        self.sweep_period_label.setVisible(is_chirp)
        self.sweep_period_edit.setVisible(is_chirp)

    # -------------------------
    # 参数读取
    # -------------------------
    def _get_common_params(self):
        fs = safe_eval_float(self.fs_edit.text(), "采样率 fs")
        duration = safe_eval_float(self.duration_edit.text(), "持续时间")
        freq_offset = safe_eval_float(self.freq_offset_edit.text(), "频偏")
        power = safe_eval_float(self.power_edit.text(), "干扰功率")
        phase0 = safe_eval_float(self.phase0_edit.text(), "初始相位")
        noise_power = safe_eval_float(self.noise_power_edit.text(), "热噪声功率")

        return fs, duration, freq_offset, power, phase0, noise_power

    # -------------------------
    # 生成信号
    # -------------------------
    def _generate_signal(self):
        fs, duration, freq_offset, power, phase0, noise_power = self._get_common_params()
        mode = self.type_combo.currentText()

        if mode == "单音干扰":
            interference = single_tone_interference(
                fs=fs,
                freq_offset=freq_offset,
                duration=duration,
                power=power,
                phase0=phase0
            )
        elif mode == "窄带干扰":
            bandwidth = safe_eval_float(self.bandwidth_edit.text(), "带宽")
            interference = narrowband_interference(
                fs=fs,
                freq_offset=freq_offset,
                bandwidth=bandwidth,
                duration=duration,
                power=power,
                phase0=phase0
            )
        elif mode == "线性扫频干扰":
            sweep_bw = safe_eval_float(self.sweep_bw_edit.text(), "扫频带宽")
            sweep_period = safe_eval_float(self.sweep_period_edit.text(), "扫频周期")
            interference = linear_chirp_interference(
                fs=fs,
                freq_offset=freq_offset,
                sweep_bandwidth=sweep_bw,
                duration=duration,
                sweep_period=sweep_period,
                power=power,
                phase0=phase0
            )
        else:
            raise ValueError("未知干扰类型")

        noise = complex_awgn(len(interference), noise_power=noise_power)
        signal = interference + noise

        self.interference = interference
        self.noise = noise
        self.signal = signal
        self.fs = fs

        total_power = np.mean(np.abs(signal) ** 2)
        interf_power = np.mean(np.abs(interference) ** 2)
        real_noise_power = np.mean(np.abs(noise) ** 2)
        jnr_db = 10 * np.log10(interf_power / (real_noise_power + 1e-12) + 1e-12)

        self.status_label.setText(
            f"状态：已生成\n"
            f"类型：{mode}\n"
            f"样本数：{len(signal)}\n"
            f"fs：{fs:.6g} Hz\n"
            f"干扰功率：{interf_power:.6g}\n"
            f"噪声功率：{real_noise_power:.6g}\n"
            f"总功率：{total_power:.6g}\n"
            f"J/N：{jnr_db:.3f} dB"
        )

    # -------------------------
    # 绘图
    # -------------------------
    def _plot_signal(self):
        if self.signal is None:
            return

        display_mode = self.display_combo.currentText()
        view_samples = safe_eval_int(self.view_samples_edit.text(), "显示样本点数")
        fs = self.fs

        x = self.signal
        n = min(view_samples, len(x))
        x_view = x[:n]

        ax = self.canvas.ax
        fig = self.canvas.fig
        ax.clear()
        ax.grid(True, alpha=0.3)

        # 清理可能残留的 colorbar
        if hasattr(self, "_cbar") and self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        if display_mode == "时域波形":
            t_ms = np.arange(n) / fs * 1e3
            ax.plot(t_ms, np.real(x_view), label="I路(实部)", linewidth=1.0)
            ax.plot(t_ms, np.imag(x_view), label="Q路(虚部)", linewidth=1.0, alpha=0.85)
            ax.set_title("时域波形")
            ax.set_xlabel("时间 (ms)")
            ax.set_ylabel("幅度")
            ax.legend()

        elif display_mode == "频谱":
            f, mag_db = compute_spectrum(x_view, fs)
            ax.plot(f / 1e6, mag_db, linewidth=1.0)
            ax.set_title("频谱")
            ax.set_xlabel("频率 (MHz)")
            ax.set_ylabel("幅度 (dB)")
            if len(f) > 0:
                ax.set_xlim(np.min(f) / 1e6, np.max(f) / 1e6)

        elif display_mode == "时频图":
            nperseg = safe_eval_int(self.nperseg_edit.text(), "时频窗长")
            noverlap = safe_eval_int(self.noverlap_edit.text(), "时频重叠")

            f, t, Sxx_db = compute_spectrogram_iq(
                x_view,
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap
            )

            if Sxx_db.size == 0:
                raise ValueError("样本过短，无法生成时频图，请增大持续时间或显示样本点数。")

            im = ax.pcolormesh(
                t * 1e3,
                f / 1e6,
                Sxx_db,
                shading="gouraud"
            )
            ax.set_title("时频图")
            ax.set_xlabel("时间 (ms)")
            ax.set_ylabel("频率 (MHz)")
            self._cbar = fig.colorbar(im, ax=ax)
            self._cbar.set_label("功率谱密度 (dB)")

        else:
            raise ValueError("未知显示模式")

        self.canvas.draw()

    # -------------------------
    # 主生成流程
    # -------------------------
    def generate_and_plot(self):
        try:
            self._generate_signal()
            self._plot_signal()
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    # -------------------------
    # 清空
    # -------------------------
    def clear_plot(self):
        self.signal = None
        self.interference = None
        self.noise = None

        ax = self.canvas.ax
        ax.clear()
        ax.set_title("信号显示")
        ax.set_xlabel("横轴")
        ax.set_ylabel("幅度 / dB")
        ax.grid(True, alpha=0.3)

        if hasattr(self, "_cbar") and self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

        self.canvas.draw()
        self.status_label.setText("状态：已清空")

    # -------------------------
    # 自动刷新
    # -------------------------
    def _toggle_auto_refresh(self):
        if self.auto_refresh_check.isChecked():
            self.timer.start(self.refresh_ms_spin.value())
        else:
            self.timer.stop()

    def stop_auto_refresh(self):
        self.auto_refresh_check.setChecked(False)
        self.timer.stop()

    # -------------------------
    # 保存信号
    # -------------------------
    def save_signal(self, fmt: str):
        if self.signal is None:
            QMessageBox.warning(self, "提示", "请先生成信号。")
            return

        try:
            if fmt == "npy":
                path, _ = QFileDialog.getSaveFileName(
                    self, "保存为 NPY", "gnss_interference.npy", "NumPy Files (*.npy)"
                )
                if not path:
                    return
                np.save(path, self.signal)

            elif fmt == "bin":
                path, _ = QFileDialog.getSaveFileName(
                    self, "保存为 BIN", "gnss_interference.bin", "Binary Files (*.bin)"
                )
                if not path:
                    return
                self.signal.astype(np.complex64).tofile(path)

            elif fmt == "mat":
                path, _ = QFileDialog.getSaveFileName(
                    self, "保存为 MAT", "gnss_interference.mat", "MAT Files (*.mat)"
                )
                if not path:
                    return
                savemat(path, {
                    "signal": self.signal,
                    "interference": self.interference,
                    "noise": self.noise,
                    "fs": self.fs
                })

            else:
                raise ValueError("不支持的保存格式")

            QMessageBox.information(self, "保存成功", f"文件已保存：\n{path}")

        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))


def main():
    app = QApplication(sys.argv)
    win = GNSSInterferenceProGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
