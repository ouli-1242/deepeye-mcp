# DeepEye 改进方案 A 设计文档

日期：2026-08-08
状态：已批准（用户确认"最优解"）

## 背景与目标

DeepEye 是给纯文本大模型补视觉能力的 MCP Server。本次改进目标是让 MCP 功能更好用，核心约束：**只做文本模型没有的能力**（视觉理解和让 MCP 更快更稳的服务端工程），不复制文本模型已有的能力（纯文本 PDF 提取、文本转 Markdown 等）。

本次范围：两个纯视觉新工具 + 错误分类提示。缓存持久化、多后端故障切换留待下轮。

## 约束

- 全部图像输入复用现有三种来源（本地路径 / http(s) URL / data URI），统一走 `parse_image_source` + `preprocess_image`
- 视觉模型只承担"读图"这一核心价值；格式转换、汇总等确定性工作由服务端完成
- 不引入新依赖
- 所有工具统一走错误分类

## 功能 1：`extract_table` — 图片/截图转表格

### 目标
覆盖四类表格：截图数据表、纸质表格、图表转数据、复杂表格（合并单元格）。

### 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `image_source` | string | 是 | 三种来源之一 |
| `model` | string | 否 | 临时指定视觉模型 |

### 数据流

```
图片 → (视觉模型) → JSON {"columns":N, "rows":[{"cells":[{"text","rowspan","colspan"}]}]}
        ↓
服务端转换：
  1. 始终转 Markdown 表格
  2. 任何 cell 存在 rowspan/colspan > 1 → 追加 JSON 元数据
```

### 关键设计

- **提示词**：要求模型输出指定 JSON 结构，突出两点——① 完整保留所有单元格数据，绝不丢数据（数据完整性优先于格式）；② 合并单元格用 `rowspan`/`colspan` 标注。
- **服务端转换**：Markdown 转换是纯函数（`json_to_markdown`），确定性、可单测。
- **复杂表格**：检测到任一 span > 1 时，输出 = Markdown + 附加 JSON（含 span 信息），供调用方按需使用。
- **容错**：模型未返回合法 JSON → 重试（复用 `_LAYOUT_RETRIES` 模式）；重试后仍失败 → 降级返回原始文本。

### 依赖

- `tools.py` 新增 `extract_table`，复用 `_run_vision`（`use_cache=False`，布局类输出不稳定）
- 新增 `table.py`：`json_to_markdown(parsed) -> str` 纯函数 + 提示词常量
- `server.py` 注册工具

## 功能 2：`analyze_images` — 批量图片分析

### 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `image_sources` | string[] | 是 | 多张图片来源（数组） |
| `prompt` | string | 否 | 应用于每张图的统一提示词，默认详细描述 |
| `model` | string | 否 | 临时指定视觉模型 |

### 数据流

```
image_sources ──并发遍历──> 每张 _run_vision（视觉）→ 逐图结果列表
        ↓
逐图描述文本拼接 ──> 一次纯文本请求（同一 prompt + "对比总结"指令）→ 跨图汇总
```

### 关键设计

- **并发**：`asyncio.gather` 并行处理每张图，单张失败不阻塞整体（返回该张的错误占位）。
- **汇总用纯文本合成**：把逐图描述作为文本发给模型总结对比。所有后端都可处理纯文本请求，无多图兼容性风险，不新增失败点。
- **输出格式**：
  ```
  [1] {来源名}: {描述}
  [2] {来源名}: {描述}

  【跨图汇总】
  {对比总结}
  ```
- **来源名**：从 `image_source` 提取（本地路径取文件名，URL 取最后一段，data URI 用序号）。

### 依赖

- `tools.py` 新增 `analyze_images`
- 单图错误经 `classify_error` 分类后作为占位文本

## 功能 3：错误分类提示

### 目标
把现有千篇一律的 `图片分析失败：{exc}` 改为按类型分类、带可操作建议的中文提示。

### 设计

新增 `errors.py`，核心为纯函数：

```
classify_error(exc: Exception, provider: str) -> tuple[str, str]
# 返回 (分类, 用户提示语)
```

| 分类 | 触发特征 | 提示语要点 |
|------|----------|-----------|
| `config` | 缺 API key / base_url / provider 无效 / `ValueError("...未配置")` | 指明缺哪个配置、在哪设置 |
| `network` | `httpx.ConnectError`、DNS 失败 | 检查网络或 BASE_URL |
| `timeout` | `httpx.TimeoutException` | 建议增大 REQUEST_TIMEOUT |
| `backend` | `httpx.HTTPStatusError`（按 400/401/403/429/5xx 细分） | 400→参数（如 response_format 不支持）；401/403→Key 无效；429→限流稍后重试 |
| `image` | `FileNotFoundError`、SSRF `ValueError`、大小超限 `ValueError` | 文件不存在 / SSRF 拦截 / 超大小 |
| `unknown` | 其他 | 保留原始异常文本 |

判断依据：异常类型为主，结合消息关键字（如 SSRF 防护、未配置、大小上限）。

### 接入

`tools.py` 内 6 个工具（现有 4 个 + 新增 2 个）的 `except Exception` 分支统一改为：

```python
content = [TextContent(type="text", text=classify_error(exc, settings.vision_provider)[1])]
```

## 测试计划

- **extract_table**
  - mock 适配器返回标准 JSON → 验证 Markdown 输出正确
  - mock 返回含 rowspan/colspan>1 → 验证追加 JSON
  - mock 返回 "说明文字+JSON" → 验证 `_extract_json` 提取
  - mock 返回无效 JSON → 验证重试 + 降级提示
  - `json_to_markdown` 纯函数单测（普通表 / 合并单元格 / 空数据）
- **analyze_images**
  - 两张图并发 → 验证逐图结果 + 汇总请求只发一次且为纯文本
  - 单张失败 → 验证占位错误、不阻塞整体
  - 空数组 → 验证友好报错
- **errors**
  - `classify_error` 各分类单测（构造各类异常）
  - 6 个工具的错误路径集成测试

## 范围外（下轮讨论）

- 缓存持久化（磁盘）
- 多后端故障切换
- 多图单请求（放弃，理由见设计：兼容性 + token 成本）

## 文件改动清单

| 文件 | 改动 |
|------|------|
| `src/deepeye_mcp/errors.py` | 新增：`classify_error` |
| `src/deepeye_mcp/table.py` | 新增：`json_to_markdown` + 提示词 |
| `src/deepeye_mcp/tools.py` | 新增 `extract_table`、`analyze_images`；错误路径接入分类 |
| `src/deepeye_mcp/server.py` | 注册 2 个新工具 |
| `tests/` | 新增 `test_errors.py`、`test_table.py`；扩 `test_tools.py` |
| `README.md` | 工具一览加 2 个新工具 |