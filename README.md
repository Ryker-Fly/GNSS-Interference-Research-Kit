# GIRK: GNSS Interference Research Kit  
# GIRK：GNSS 干扰研究工具包

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/github/license/Ryker-Fly/GNSS-Interference-Research-Kit)](./LICENSE)


A Python toolkit for simulating and mitigating GNSS (Global Navigation Satellite System) interference, including single-tone, narrowband interference generation, and advanced mitigation techniques such as FDPB (Frequency Domain Pulse Blanking) and IIR notch filters.

Designed for researchers and engineers working on GNSS signal integrity, this package provides modular, well-documented components to accelerate interference analysis and countermeasure development.

一个用于生成与抑制 GNSS（全球导航卫星系统）干扰的 Python 工具包，支持单音干扰、窄带干扰、扫频干扰等干扰生成，以及 FDPB（频域脉冲消隐）和 IIR 陷波滤波等先进抗干扰算法。

本工具包面向从事 GNSS 信号完整性研究的科研人员与工程师，提供模块化、文档完善的组件，助力干扰分析与抗干扰技术的快速开发。

---

## 📦 Features / 功能特性

- **Interference Generation / 干扰信号生成**:
  - Single-tone interference / 单音干扰
  - Narrowband interference with configurable bandwidth and center frequency / 可配置带宽与中心频率的窄带干扰
  - Linear Chirp interference / 线性扫频干扰

- **Mitigation Algorithms / 抗干扰算法**:
  - FDPB (Frequency Domain Pulse Blanking) / 频域脉冲消隐（FDPB）
  - IIR Notch filter-based mitigation / 基于 IIR 陷波滤波的抑制方法

- **Ready-to-Run Examples / 开箱即用示例**:
  - Jupyter Notebooks for interactive exploration / 支持交互式探索的 Jupyter Notebook
  - Standalone Python scripts for automation / 适用于自动化的独立 Python 脚本

- **Modular Design / 模块化设计**:
  - Clean separation between signal generation, interference injection, and mitigation / 信号生成、干扰注入与抗干扰模块清晰分离
  - Easy to extend with new interference types or algorithms / 易于扩展新的干扰类型或算法

---

## 💎 Advanced Features (Pro Version) / 高级功能（Pro 版本）

For **threshold-free FDPB**, real-time processing, and commercial support,  
please contact the author: **ryker_fly@126.com**

如需 **无门限自适应 FDPB**、实时处理能力或商业技术支持，  
请联系作者：**ryker_fly@126.com**

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- `git` (to clone the repository)

### Steps
```bash
# Clone the repository
git clone https://github.com/Ryker-Fly/GNSS-Interference-Research-Kit.git
cd GNSS-Interference-Research-Kit

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/macOS
# or
venv\Scripts\activate        # Windows

# Install the package in editable mode
pip install -e .

## ▶️ Quick Start

### Run a Python Script
```bash
python examples/demo.py

### Execute a Jupyter Notebook (without opening browser)
jupyter nbconvert --to notebook --execute examples/01_single_tone_interference.ipynb

## Project Structure
GNSS-Interference-Research-Kit/
├── girk/                     # Core package
│   ├── __init__.py
│   ├── interference/         # Interference generators
│   └── mitigation/           # Mitigation algorithms
├── examples/                 # Example scripts and notebooks
│   ├── demo.py               # Minimal working example
│   ├── 01_single_tone_interference.ipynb
│   ├── 02_narrowband_interference.ipynb
│   ├── 03_linear_chirp_interference.ipynb
│   ├── 11_fdpb_mitigation_demo.ipynb
│   └── 12_iir_notch_mitigation_demo.ipynb
├── requirements.txt          # Exact dependency versions
├── setup.py                  # Package metadata and dependencies
├── .gitignore                # Files excluded from version control
├── LICENSE                   # MIT License
└── README.md                 # This file
