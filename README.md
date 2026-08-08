# DeepEye

为纯文本大模型提供视觉能力的 MCP Server（图像描述 / OCR / 视觉问答 / 布局分析）。

## 安装

```bash
pip install deepeye-mcp
```

或从源码安装（开发者）：

**第 1 步：下载源码并进入目录**

```bash
git clone https://github.com/ouli-1242/deepeye-mcp.git
cd deepeye-mcp
```

> 注意：`cd` 进入的是克隆下来的 `deepeye-mcp` 目录（即你刚才下载的文件夹名）。

**第 2 步：创建虚拟环境**

```bash
python -m venv .venv
```

**第 3 步：激活虚拟环境 + 安装**

Windows：

```bash
.venv\Scripts\activate
pip install -e .
```

macOS / Linux：

```bash
source .venv/bin/activate
pip install -e .
```

装好后命令 `deepeye` 即可用。

要求：Python 3.11+，一个视觉模型 API Key。

## 卸载

```bash
pip uninstall deepeye-mcp
```

若装在虚拟环境中，先激活再卸载：

```bash
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS / Linux
pip uninstall deepeye-mcp
```

## 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`：

```dotenv
VISION_PROVIDER=openai
OPENAI_API_KEY=sk-your-real-key-here
OPENAI_MODEL=gpt-5.6-luna
# 如果用兼容服务，可改 OPENAI_BASE_URL
# OPENAI_BASE_URL=https://your-compatible-service/v1
```

## 启动

```bash
deepeye
```

Server 通过 stdio 与 MCP 客户端通信，单独运行不会输出交互界面，需配合 MCP 客户端使用（见 [MCP 客户端集成](#mcp-客户端集成)）。

## 工具一览

### `describe_image` — 通用图像理解

对图片进行详细描述，可自定义描述角度。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `image_source` | string | 是 | 本地路径 / http(s) URL / `data:image/...;base64,...` |
| `prompt` | string | 否 | 描述提示词，不传则使用默认详细描述 |
| `model` | string | 否 | 临时指定视觉模型，不传则用配置默认值 |

**返回**：`图片分析结果：\n{描述}`

### `extract_text` — OCR 文字提取

仅提取图片中的文字，保持原文排版，不加任何额外描述。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `image_source` | string | 是 | 同上 |
| `language` | string | 否 | 识别语言，`auto`（默认）自动识别；其他值如 `zh` / `en` 会附加语言提示 |

**返回**：图片中提取到的纯文字。

### `ask_about_image` — 视觉问答

针对图片内容提出具体问题，获取定向回答。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `image_source` | string | 是 | 同上 |
| `question` | string | 是 | 要询问的问题 |

**返回**：针对问题的回答。

### `analyze_layout` — UI 布局结构化分析

对图片进行 UI 布局结构化分析，返回 JSON 格式的布局类型与元素树（类型/位置/样式），适合前端复刻。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `image_source` | string | 是 | 同上 |
| `detail` | string | 否 | 分析粒度：`basic`（默认，仅类型 + 文本 + 位置）或 `detailed`（额外返回颜色、字号、圆角等样式） |
| `model` | string | 否 | 临时指定视觉模型，不传则用配置默认值 |

**返回**：JSON 字符串，包含 `layout_type`（布局类型）、`summary`（一句话描述）与 `elements`（元素树）；每个元素含 `type`、`text`、`position`（百分比坐标）、`children`，`detailed` 模式额外返回 `styles`。

## 配置参考

所有配置通过环境变量或 `.env` 文件加载（参考 `.env.example`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VISION_PROVIDER` | `openai` | 视觉后端提供者：`openai` / `gemini` / `custom` |
| `OPENAI_API_KEY` | — | OpenAI 或兼容服务的 API Key |
| `OPENAI_MODEL` | `gpt-5.6-luna` | 视觉模型名称 |
| `OPENAI_BASE_URL` | — | 接口地址，留空用官方 `https://api.openai.com/v1`；可改为 Azure / 代理 / 兼容服务 |
| `GEMINI_API_KEY` | — | Gemini 后端 API Key |
| `GEMINI_MODEL` | `gemini-1.5-pro` | Gemini 模型名称 |
| `CUSTOM_API_KEY` | — | 自定义 OpenAI 兼容服务 Key |
| `CUSTOM_BASE_URL` | — | 自定义服务接口地址 |
| `CUSTOM_MODEL` | `qwen-vl-max` | 自定义模型名称 |
| `OCR_BACKEND` | `openai` | `extract_text` 实际使用的视觉后端 |
| `IMAGE_MAX_DIM` | `2048` | 图片预处理最大边长（像素），超过则等比缩放转 JPEG；`0` 禁用预处理 |
| `CACHE_ENABLED` | `true` | 是否开启视觉结果缓存（LRU + TTL） |
| `CACHE_MAX_SIZE` | `128` | 缓存最大条目数 |
| `CACHE_TTL` | `3600` | 缓存存活秒数 |
| `REQUEST_TIMEOUT` | `120` | 视觉后端 HTTP 请求超时（秒） |
| `MAX_RETRIES` | `3` | 失败重试次数（仅对网络/超时错误重试） |
| `MAX_TOKENS` | `4096` | 视觉模型返回的最大 token 数 |
| `REASONING_EFFORT` | 空 | 推理深度 `low`/`medium`/`high`；留空不发送该参数（部分后端不支持） |
| `MAX_IMAGE_BYTES` | `20971520` | URL 图片下载大小上限（字节），超过拒绝 |
| `ALLOW_PRIVATE_URLS` | `false` | 是否允许访问内网/保留地址（SSRF 防护，默认禁止；仅本地调试设为 `true`） |

用兼容服务的例子（阿里通义 Qwen-VL）：

```dotenv
VISION_PROVIDER=openai
OPENAI_API_KEY=sk-your-dashscope-key
OPENAI_MODEL=qwen-vl-max
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

切换到 Gemini：

```bash
VISION_PROVIDER=gemini
GEMINI_API_KEY=你的key
GEMINI_MODEL=gemini-2.0-flash
```

## MCP 客户端集成

DeepEye 是标准 stdio MCP Server，在 MCP 配置中声明 `deepeye` 启动命令，并通过 `env` 字段传入视觉后端凭证。

### 方式一：Claude Code 命令行（推荐）

```bash
claude mcp add deepeye -- /path/to/deepeye/.venv/bin/deepeye \
  --env VISION_PROVIDER=openai \
  --env OPENAI_API_KEY=sk-your-key \
  --env OPENAI_MODEL=gpt-5.6-luna
```

Windows 下把命令路径换成 `.venv\Scripts\deepeye.exe`。

### 方式二：配置文件 `.mcp.json`

在项目根目录（或 Claude Code 起始目录）创建 `.mcp.json`：

```json
{
  "mcpServers": {
    "deepeye": {
      "command": "D:/tools/deepeye/.venv/Scripts/deepeye.exe",
      "env": {
        "VISION_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-your-key",
        "OPENAI_BASE_URL": "https://your-compatible-service/v1",
        "OPENAI_MODEL": "gpt-5.6-luna"
      }
    }
  }
}
```

保存后重启 Claude Code，`/mcp` 面板中应显示 `deepeye` 已连接。若显示 failed，运行 `.venv/Scripts/deepeye.exe` 查看报错。

其他客户端（Cursor / Cline / Windsurf 等）的配置方式见 [接入 Coding Agent 指南](docs/coding-agent-integration.md)。

> **opencode 用户**：安装 [opencode-easy-vision](https://github.com/devadathanmb/opencode-easy-vision) 插件后，粘贴图片会自动保存为临时文件并调用 DeepEye 分析。配置方法见 [接入指南的 opencode 章节](docs/coding-agent-integration.md#进阶粘贴图片自动调用-deepeyeopencode-easy-vision-插件)。

## 支持的视觉后端

| 后端 | 状态 | 说明 |
|------|------|------|
| **OpenAI 兼容** | 已实现 | 支持 OpenAI 官方、Azure OpenAI、阿里通义 Qwen-VL、智谱 GLM-4V、Moonshot 等 |
| **Gemini** | 已实现 | 支持 Google Gemini 系列模型（gemini-1.5-pro / gemini-2.0-flash 等） |
| **自定义 OpenAI 兼容** | 已实现 | 用于任何兼容 OpenAI Chat Completions 格式的自部署服务（vLLM / Ollama / 通义 Qwen-VL / 智谱等） |
| 本地 OCR (Tesseract / PaddleOCR) | 计划中 | 隐私场景下数据不出本机 |

## 开发

### 运行测试

```bash
pytest tests/ -v
```

测试覆盖图像源解析、视觉适配器工厂、四个工具的 prompt 组装逻辑，全部使用 mock，不发起真实 API 调用。

### 新增视觉后端

1. 在 `src/deepeye_mcp/vision/` 下新增 `xxx_adapter.py`，继承 `VisionAdapter`，实现 `describe` 方法
2. 在 `vision/__init__.py` 的工厂函数中注册新分支

## 项目结构

```
deepeye/
├── pyproject.toml              # 项目元数据、依赖、入口命令、pytest 配置
├── .env.example                # 配置示例
├── src/
│   └── deepeye_mcp/
│       ├── __init__.py         # __version__
│       ├── server.py           # MCP Server 组装（mcp 2.0 API）
│       ├── tools.py            # 四个 MCP 工具实现
│       ├── image_utils.py      # 图像源解析（本地/URL/data URI）
│       ├── config.py           # pydantic-settings 配置加载
│       ├── cache.py            # 视觉结果缓存
│       └── vision/
│           ├── __init__.py     # create_vision_adapter 工厂
│           ├── base.py         # VisionAdapter 抽象基类
│           ├── openai_adapter.py
│           ├── gemini_adapter.py
│           └── custom_adapter.py
└── tests/
    ├── test_image_utils.py
    ├── test_vision_factory.py
    ├── test_adapters.py
    └── test_tools.py
```

## License

[MIT](LICENSE) © DeepEye Contributors