import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget


class GnssMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GIRK - GNSS抗干扰评估工具")

        # 创建布局
        layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # 创建绘图控件 (用于显示频谱)
        self.plot_widget = pg.PlotWidget(title="实时频谱分析")
        layout.addWidget(self.plot_widget)

        # 模拟生成一段带干扰的信号频谱
        self.curve = self.plot_widget.plot(pen='y')
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # 50ms 刷新一次

    def update_plot(self):
        # 模拟 2048 点的 FFT 结果
        data = np.random.normal(size=2048) + np.sin(np.linspace(0, 100, 2048)) * 10
        self.curve.setData(data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GnssMonitor()
    window.show()
    sys.exit(app.exec())
