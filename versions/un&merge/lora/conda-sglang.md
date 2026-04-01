# conda-sglang

## conda环境配置
你可以在终端中运行以下命令来创建名为 `sglang` 的 conda 环境：
```bash
conda create -n sglang python=3.10
```
这会创建一个包含 Python 3.10 的新环境。你可以根据需要更改 Python 版本。创建完成后，使用以下命令激活环境：
```bash
conda activate sglang
```

## 安装依赖
### Use the last release branch
git clone -b v0.4.10 https://github.com/sgl-project/sglang.git
cd sglang

### Install the python packages
pip install --upgrade pip
pip install -e "python[all]"