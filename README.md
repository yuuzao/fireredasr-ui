# FireRedASR-UI

一个用于 [FireRedASR](https://github.com/FireRedTeam/FireRedASR) 的 WebUI 及 API 项目，API 兼容 OpenAI 格式。

FireRedASR 是一个高度精确的中文语音转文字项目，支持 AED 和 LLM 两种识别模式。

## ✨ 功能特性

- 🎯 **高精度中文语音识别**：基于 FireRedASR 模型，支持 AED 和 LLM 两种识别模式
- 🌐 **WebUI 界面**：友好的图形界面，支持拖拽上传和实时识别
- 🔌 **OpenAI 兼容 API**：完全兼容 OpenAI 语音识别 API 格式
- 📝 **智能标点恢复**：支持 FireRedChat-punc 模型和基于规则的标点恢复
- 🐳 **Docker 支持**：提供 Docker 和 Docker Compose 部署方式
- 📊 **SRT 字幕格式**：支持生成 SRT 格式的字幕文件

## 🖼️ WebUI 预览

![](./static/ui0.png)

## 🚀 快速开始

### Docker 部署（推荐）

1. **克隆仓库**
   ```bash
   git clone https://github.com/jianchang512/fireredasr-ui.git
   cd fireredasr-ui
   ```

2. **下载模型**
   - 按照 [模型下载](#模型下载) 章节下载所需模型
   - 将模型文件放入 `pretrained_models` 目录

3. **启动服务**
   ```bash
   docker-compose up -d
   ```

4. **访问 WebUI**
   - 浏览器打开：http://localhost:35078

### 源码安装（Linux/MacOS）

#### 环境要求

- Python 3.10+
- FFmpeg
- CUDA（可选，用于 GPU 加速）

#### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/jianchang512/fireredasr-ui.git
   cd fireredasr-ui
   ```

2. **创建虚拟环境**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/MacOS
   # 或
   . venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **下载模型**
   - 按照 [模型下载](#模型下载) 章节下载所需模型

5. **启动服务**
   ```bash
   python app.py
   ```

6. **访问 WebUI**
   - 浏览器打开：http://127.0.0.1:5078

### Windows 整合包

> ⚠️ **注意**：模型文件较大（约 21G），整合包仅包含程序主体，需要单独下载模型。

1. **下载整合包**
   - 下载地址：https://github.com/jianchang512/fireredasr-ui/releases/download/v0.3/fireredASR-2025-0224.7z
   - 解压到任意目录

2. **下载模型**
   - 按照 [模型下载](#模型下载) 章节下载模型
   - 将模型文件放入 `pretrained_models` 目录

3. **启动服务**
   - 双击 `启动.bat` 文件
   - 浏览器自动打开：http://127.0.0.1:5078

## 📦 模型下载

### 必需模型

#### 1. FireRedASR-AED-L 模型

- **下载地址**：[HuggingFace](https://huggingface.co/FireRedTeam/FireRedASR-AED-L/tree/main)
- **文件大小**：约 4.35G
- **存放位置**：`pretrained_models/FireRedASR-AED-L/`
- **必需文件**：
  - `model.pth.tar`
  - `config.yaml`
  - `cmvn.txt`
  - `dict.txt`
  - `train_bpe1000.model`

#### 2. FireRedASR-LLM-L 模型

- **下载地址**：[HuggingFace](https://huggingface.co/FireRedTeam/FireRedASR-LLM-L/tree/main)
- **文件大小**：约 3.37G
- **存放位置**：`pretrained_models/FireRedASR-LLM-L/`
- **必需文件**：
  - `model.pth.tar`
  - `asr_encoder.pth.tar`
  - 其他配置文件

#### 3. Qwen2-7B-Instruct 模型

- **下载地址**：[HuggingFace](https://huggingface.co/Qwen/Qwen2-7B-Instruct/tree/main)
- **文件大小**：约 17G（4 个文件）
- **存放位置**：`pretrained_models/FireRedASR-LLM-L/Qwen2-7B-Instruct/`
- **必需文件**：
  - `model-00001-of-00004.safetensors`
  - `model-00002-of-00004.safetensors`
  - `model-00003-of-00004.safetensors`
  - `model-00004-of-00004.safetensors`
  - 其他配置文件

### 可选模型

#### FireRedChat-punc 标点恢复模型

- **下载地址**：[ModelScope](https://www.modelscope.cn/models/FireRedTeam/FireRedChat-punc)
- **存放位置**：`pretrained_models/FireRedChat-punc/`
- **必需文件**：
  - `model.pth.tar`
  - `chinese-lert-base/` 目录（需要单独下载 `hfl/chinese-lert-base` 模型）

> 💡 **提示**：如果未下载标点恢复模型，系统会自动使用基于规则的标点恢复方法。

### 模型下载说明

> ⚠️ **重要提示**：
> - HuggingFace 网站在国内无法直接访问，需要使用代理或镜像站点
> - 所有模型文件合计约 21G，请确保有足够的磁盘空间
> - 建议使用支持断点续传的下载工具

## 📝 标点恢复功能

本项目支持两种标点恢复方式：

### 1. FireRedChat-punc 模型（推荐）

使用 FireRedTeam 提供的标点恢复模型，能够更准确地恢复标点符号。

**特点**：
- ✅ 基于深度学习的标点恢复
- ✅ 支持中文标点：，。？！等
- ✅ 自动回退机制：模型不可用时自动使用规则方法

**配置**：
- 模型路径：`pretrained_models/FireRedChat-punc/`
- 基础模型：需要下载 `hfl/chinese-lert-base` 到 `chinese-lert-base/` 子目录
- 环境变量：可通过 `PUNCTUATION_MODEL_PATH` 指定自定义路径

### 2. 基于规则的标点恢复

当 FireRedChat-punc 模型不可用时，系统会自动使用基于规则的标点恢复方法。

**特点**：
- ✅ 无需额外模型文件
- ✅ 根据文本特征和时间间隔添加标点
- ✅ 支持疑问词和感叹词识别

## 🔌 API 使用

### API 地址

- **默认地址**：http://127.0.0.1:5078/v1
- **Docker 部署**：http://localhost:35078/v1

### OpenAI SDK 示例

```python
from openai import OpenAI

client = OpenAI(
    api_key='123456',
    base_url='http://127.0.0.1:5078/v1'
)

audio_file = open("audio.wav", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="json",  # 或 "srt"
    timeout=86400
)

print(transcript.text)
```

### cURL 示例

```bash
curl -X POST http://127.0.0.1:5078/v1/audio/transcriptions \
  -H "Authorization: Bearer 123456" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

### 支持的参数

- `model`: 固定为 `"whisper-1"`（兼容 OpenAI 格式）
- `file`: 音频文件（支持 wav, mp3, m4a 等格式）
- `response_format`: 响应格式
  - `"json"`: JSON 格式（默认）
  - `"srt"`: SRT 字幕格式
- `language`: 语言代码（可选，默认为中文）

## 🐳 Docker 配置

### docker-compose.yml

项目提供了完整的 Docker Compose 配置，包括：

- **端口映射**：35078:5078
- **卷挂载**：
  - `./pretrained_models:/app/pretrained_models:ro` - 模型目录（只读）
  - `./logs:/app/logs` - 日志目录
  - `./static/tmp:/app/static/tmp` - 临时文件目录

### 自定义配置

可以通过修改 `docker-compose.yml` 文件自定义配置：

```yaml
services:
  fireredasr-ui:
    ports:
      - "8080:5078"  # 修改端口
    environment:
      - PUNCTUATION_MODEL_PATH=/custom/path  # 自定义标点模型路径
```

## 🔧 配置说明

### 环境变量

- `PUNCTUATION_MODEL_PATH`: 标点恢复模型路径（可选）

### 模型路径结构

```
pretrained_models/
├── FireRedASR-AED-L/          # AED 模型
│   ├── model.pth.tar
│   ├── config.yaml
│   └── ...
├── FireRedASR-LLM-L/          # LLM 模型
│   ├── model.pth.tar
│   ├── asr_encoder.pth.tar
│   └── Qwen2-7B-Instruct/     # Qwen 模型
│       ├── model-00001-of-00004.safetensors
│       └── ...
└── FireRedChat-punc/          # 标点恢复模型（可选）
    ├── model.pth.tar
    └── chinese-lert-base/     # 基础模型
        └── ...
```

## 🛠️ 开发

### 项目结构

```
fireredasr-ui/
├── app.py                      # Flask 应用主文件
├── fireredasr/                # 核心模块
│   ├── models/                # 模型相关
│   │   ├── fireredasr.py      # ASR 模型
│   │   └── punctuation.py     # 标点恢复模块
│   ├── tokenizer/              # 分词器
│   └── utils/                  # 工具函数
├── static/                     # 静态文件
├── templates/                  # 模板文件
├── pretrained_models/          # 模型目录
├── requirements.txt            # Python 依赖
├── Dockerfile                  # Docker 镜像构建文件
└── docker-compose.yml         # Docker Compose 配置
```

### 依赖项

主要依赖包括：
- `torch>=2.0.0` - PyTorch
- `transformers>=4.51.0` - Transformers 库
- `flask` - Web 框架
- `modelscope` - ModelScope 支持
- 其他依赖见 `requirements.txt`

## 📚 相关项目

- [FireRedASR](https://github.com/FireRedTeam/FireRedASR) - 核心 ASR 模型
- [FireRedChat-punc](https://www.modelscope.cn/models/FireRedTeam/FireRedChat-punc) - 标点恢复模型

## 🙏 致谢

- [FireRedTeam/FireRedASR](https://github.com/FireRedTeam/FireRedASR) - 提供核心 ASR 模型
- [FireRedTeam/FireRedChat-punc](https://www.modelscope.cn/models/FireRedTeam/FireRedChat-punc) - 提供标点恢复模型

## 📄 许可证

本项目遵循原项目的许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**注意**：本项目仅用于学习和研究目的，请遵守相关法律法规。
