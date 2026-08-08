# DeepEye 视觉工具增强实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 deepeye 增加两个纯视觉工具（`extract_table` 图片转表格、`analyze_images` 批量图片分析）和统一错误分类提示。

**Architecture:** 视觉模型只承担"读图"（输出指定 JSON），格式转换/汇总由服务端确定性代码完成。`errors.py` 提供纯函数 `classify_error`，6 个工具统一接入。`analyze_images` 的汇总用纯文本请求（新增适配器 `describe_text` 方法），不依赖多图支持。

**Tech Stack:** Python 3.11+，mcp 2.0，httpx，pytest + pytest-asyncio（asyncio_mode=auto）

## Global Constraints

- 不引入新依赖。
- 图像输入只用现有 `parse_image_source` + `preprocess_image`（三种来源）。
- 错误统一走 `classify_error`，消息为中文、带可操作建议。
- 工具返回 `list[TextContent]`。
- 布局/表格类输出不稳定 → `use_cache=False`。
- 运行测试：`PYTHONPATH=src python -m pytest tests/ -q --basetemp=.pytest_tmp -p no:cacheprovider`（Windows 下 pytest 默认临时目录清理会报 PermissionError，必须带 `--basetemp`）。

---

### Task 1: `errors.py` — 错误分类纯函数

**Files:**
- Create: `src/deepeye_mcp/errors.py`
- Test: `tests/test_errors.py`

**Interfaces:**
- Produces: `classify_error(exc: Exception, provider: str = "") -> tuple[str, str]`，返回 `(category, message)`；category ∈ {config, network, timeout, backend, image, unknown}

- [ ] **Step 1: 写失败测试**

```python
# tests/test_errors.py
"""classify_error 错误分类单元测试。"""
import httpx
import pytest

from deepeye_mcp.errors import classify_error


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


def test_config_missing_base_url():
    exc = ValueError("custom_base_url 未配置：使用 custom 视觉后端必须设置 CUSTOM_BASE_URL")
    category, message = classify_error(exc, "custom")
    assert category == "config"
    assert "配置错误" in message


def test_config_invalid_provider():
    category, _ = classify_error(ValueError("不支持的视觉后端: 'foo'"))
    assert category == "config"


def test_image_file_not_found():
    category, message = classify_error(FileNotFoundError("图像文件不存在: /x.png"))
    assert category == "image"
    assert "文件不存在" in message


def test_image_ssrf_blocked():
    exc = ValueError("拒绝访问非公网地址（SSRF 防护）: localhost (127.0.0.1)")
    category, message = classify_error(exc)
    assert category == "image"
    assert "SSRF" in message


def test_image_too_large():
    category, message = classify_error(ValueError("图片超过大小上限: 100 > 10 bytes"))
    assert category == "image"
    assert "大小上限" in message


def test_timeout():
    category, message = classify_error(httpx.TimeoutException("timed out"))
    assert category == "timeout"
    assert "REQUEST_TIMEOUT" in message


def test_network():
    category, message = classify_error(httpx.ConnectError("refused"))
    assert category == "network"
    assert "网络错误" in message


def test_backend_400():
    category, message = classify_error(_http_error(400))
    assert category == "backend"
    assert "400" in message


def test_backend_401():
    category, message = classify_error(_http_error(401))
    assert category == "backend"
    assert "401" in message


def test_backend_429():
    category, message = classify_error(_http_error(429))
    assert category == "backend"
    assert "429" in message


def test_backend_500():
    category, message = classify_error(_http_error(500))
    assert category == "backend"
    assert "500" in message


def test_unknown():
    category, message = classify_error(RuntimeError("boom"))
    assert category == "unknown"
    assert "boom" in message


def test_openai_adapter_runtime_error_timeout():
    category, message = classify_error(RuntimeError("视觉模型请求超时（120s），已重试 3 次。"))
    assert category == "timeout"
    assert "超时" in message
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=src python -m pytest tests/test_errors.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'deepeye_mcp.errors'`

- [ ] **Step 3: 实现**

```python
# src/deepeye_mcp/errors.py
"""DeepEye 错误分类。

把底层异常映射为「分类 + 用户友好的中文提示」，供所有工具统一使用。
"""

from __future__ import annotations

import httpx


def classify_error(exc: Exception, provider: str = "") -> tuple[str, str]:
    """把异常分类为 ``(category, message)``。

    Args:
        exc: 捕获到的异常。
        provider: 当前视觉后端（openai/gemini/custom），预留用于针对性提示。

    Returns:
        ``(category, message)``；category 取值：
        config / network / timeout / backend / image / unknown。
    """
    msg = str(exc)

    # 网络 / 超时（httpx 具体异常优先）
    if isinstance(exc, httpx.TimeoutException):
        return "timeout", f"请求超时：{msg}。可增大 REQUEST_TIMEOUT 或换更快的端点。"
    if isinstance(exc, httpx.TransportError):
        return "network", f"网络错误：{msg}。请检查网络连接和 BASE_URL 配置。"

    # 后端 HTTP 错误
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 400:
            return "backend", f"后端拒绝请求（400）：{msg}。可能原因：模型不支持某参数（如 reasoning_effort / response_format）。"
        if status in (401, 403):
            return "backend", f"后端认证失败（{status}）：{msg}。API Key 可能无效。"
        if status == 404:
            return "backend", f"后端资源不存在（404）：{msg}。请检查模型名或 BASE_URL。"
        if status == 429:
            return "backend", f"后端限流（429）：{msg}。请稍后重试或降低调用频率。"
        if status >= 500:
            return "backend", f"后端服务错误（{status}）：{msg}。请稍后重试。"
        return "backend", f"后端返回错误（{status}）：{msg}"

    # 图片类
    if isinstance(exc, FileNotFoundError):
        return "image", f"图片文件不存在：{msg}"
    if isinstance(exc, ValueError):
        if "SSRF" in msg:
            return "image", f"图片来源被安全策略拦截：{msg}"
        if "大小上限" in msg:
            return "image", f"图片超过大小上限：{msg}"
        if any(kw in msg for kw in ("未设置", "未配置", "不支持的", "缺少")):
            return "config", f"配置错误：{msg} 请在 MCP 配置的 env 或 .env 中设置。"
    if isinstance(exc, KeyError):
        return "config", f"配置错误：缺少必填参数 {msg}"

    # OpenAI 适配器把 timeout/transport 包装成 RuntimeError
    if isinstance(exc, RuntimeError):
        if "超时" in msg:
            return "timeout", msg
        if "网络" in msg:
            return "network", msg

    # 未知
    return "unknown", msg
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=src python -m pytest tests/test_errors.py -q`
Expected: PASS（16 个测试）

- [ ] **Step 5: 提交**

```bash
git add tests/test_errors.py src/deepeye_mcp/errors.py
git commit -m "feat(errors): add classify_error for unified error messaging"
```

---

### Task 2: 现有 4 个工具接入错误分类

**Files:**
- Modify: `src/deepeye_mcp/tools.py`（4 个 except 分支）
- Test: `tests/test_tools.py`（新增分类断言）

**Interfaces:**
- Consumes: `classify_error`（Task 1）
- Produces: 现有工具错误消息带分类提示

- [ ] **Step 1: 写失败测试**（追加到 test_tools.py）

```python
@patch("deepeye_mcp.tools.create_vision_adapter")
async def test_describe_image_config_error_classified(mock_factory):
    """配置缺失类错误应给出「配置错误」提示。"""
    mock_adapter = MagicMock()
    mock_adapter.describe = AsyncMock(
        side_effect=ValueError("custom_base_url 未配置：使用 custom 视觉后端必须设置 CUSTOM_BASE_URL")
    )
    mock_factory.return_value = mock_adapter

    result = await describe_image(image_source=_DATA_URI)

    assert "配置错误" in result[0].text


@patch("deepeye_mcp.tools.create_vision_adapter")
async def test_analyze_layout_backend_error_has_status(mock_factory):
    """后端 429 错误应给出限流提示。"""
    import httpx

    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    mock_adapter = MagicMock()
    mock_adapter.describe = AsyncMock(
        side_effect=httpx.HTTPStatusError("限流", request=request, response=response)
    )
    mock_factory.return_value = mock_adapter

    result = await analyze_layout(image_source=_DATA_URI)

    assert "429" in result[0].text
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=src python -m pytest tests/test_tools.py -k "config_error or backend_error" -q`
Expected: FAIL（现有 handler 返回裸异常文本，无"配置错误"）

- [ ] **Step 3: 实现**（tools.py 里 4 个 except 分支）

现有：
```python
    except Exception as exc:
        return [TextContent(type="text", text=f"图片分析失败：{exc}")]
```
改为：
```python
    except Exception as exc:
        return [TextContent(type="text", text=f"图片分析失败：{classify_error(exc, settings.vision_provider)[1]}")]
```

同样改 `extract_text`（前缀 `OCR 失败：`）、`ask_about_image`（前缀 `视觉问答失败：`）、`analyze_layout`（前缀 `布局分析失败：`）。

tools.py 顶部新增导入：
```python
from deepeye_mcp.errors import classify_error
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=src python -m pytest tests/test_tools.py -q`
Expected: PASS（全部，含新增 2 个）

- [ ] **Step 5: 提交**

```bash
git add src/deepeye_mcp/tools.py tests/test_tools.py
git commit -m "feat(tools): route existing tool errors through classify_error"
```

---

### Task 3: `table.py` — 表格 JSON → Markdown 纯函数

**Files:**
- Create: `src/deepeye_mcp/table.py`
- Test: `tests/test_table.py`

**Interfaces:**
- Produces:
  - `_TABLE_JSON_PROMPT: str`（提示词常量）
  - `json_to_markdown(parsed: dict) -> str`
  - `has_merged_cells(parsed: dict) -> bool`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_table.py
"""表格 JSON → Markdown 转换单元测试。"""
from deepeye_mcp.table import has_merged_cells, json_to_markdown


def test_simple_table():
    parsed = {
        "columns": 2,
        "rows": [
            {"cells": [{"text": "名称", "rowspan": 1, "colspan": 1}, {"text": "价格"}]},
            {"cells": [{"text": "苹果", "rowspan": 1, "colspan": 1}, {"text": "5元"}]},
        ],
    }
    md = json_to_markdown(parsed)
    assert "| 名称 | 价格 |" in md
    assert "| 苹果 | 5元 |" in md
    assert "---" in md


def test_rows_ragged_padded_to_max_width():
    """合并单元格导致某行 cell 数少于最大列宽时，应补齐空列。"""
    parsed = {
        "columns": 2,
        "rows": [
            {"cells": [{"text": "A", "rowspan": 2, "colspan": 1}, {"text": "B"}]},
            {"cells": [{"text": "C"}]},
        ],
    }
    md = json_to_markdown(parsed)
    assert "| C |  |" in md


def test_merged_cells_flagged():
    parsed = {"rows": [{"cells": [{"text": "A", "rowspan": 2, "colspan": 1}]}]}
    assert has_merged_cells(parsed) is True


def test_no_merged_cells():
    parsed = {"rows": [{"cells": [{"text": "x", "rowspan": 1, "colspan": 1}]}]}
    assert has_merged_cells(parsed) is False


def test_pipe_escaped():
    parsed = {"rows": [{"cells": [{"text": "a|b"}]}]}
    md = json_to_markdown(parsed)
    assert "a\\|b" in md


def test_empty_rows():
    assert json_to_markdown({"rows": []}) == "（空表格）"
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=src python -m pytest tests/test_table.py -q`
Expected: FAIL，`ModuleNotFoundError: No module named 'deepeye_mcp.table'`

- [ ] **Step 3: 实现**

```python
# src/deepeye_mcp/table.py
"""表格提取：视觉模型 JSON → Markdown 转换。

视觉模型负责读表并输出结构化 JSON，本模块负责确定性的格式转换。
"""

_TABLE_JSON_PROMPT = """
分析这张图片中的表格，输出 JSON。只返回 JSON，不加任何说明文字。
JSON 格式：
{"columns": 列数, "title": "表格标题或空串", "rows": [{"cells": [{"text": "单元格内容", "rowspan": 1, "colspan": 1}]}]}
要求：
1. 完整保留所有单元格数据，绝不允许丢失或合并任何内容。
2. 合并单元格用 rowspan / colspan 标注数量（默认 1）。
3. 表头行作为第一行 rows 数据。
4. 图表（柱状图/折线图/饼图）转化为对应的数据表。
"""


def json_to_markdown(parsed: dict) -> str:
    """把表格 JSON 转换为 Markdown 表格。

    Args:
        parsed: 含 ``rows`` 的字典，每行含 ``cells`` 列表。

    Returns:
        Markdown 表格字符串；空表格返回 ``（空表格）``。
    """
    rows = parsed.get("rows") or []
    if not rows:
        return "（空表格）"
    max_cols = max((len(r.get("cells", [])) for r in rows), default=0)
    if max_cols == 0:
        return "（空表格）"

    def _row(cells: list) -> str:
        padded = [c.get("text", "").replace("|", "\\|") for c in cells]
        padded += [""] * (max_cols - len(padded))
        return "| " + " | ".join(padded) + " |"

    header = _row(rows[0].get("cells", []))
    sep = "|" + "---|" * max_cols
    body = [_row(r.get("cells", [])) for r in rows[1:]]
    return "\n".join([header, sep] + body)


def has_merged_cells(parsed: dict) -> bool:
    """是否存在合并单元格（rowspan/colspan > 1）。"""
    for row in parsed.get("rows") or []:
        for cell in row.get("cells", []):
            if cell.get("rowspan", 1) > 1 or cell.get("colspan", 1) > 1:
                return True
    return False
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=src python -m pytest tests/test_table.py -q`
Expected: PASS（6 个测试）

- [ ] **Step 5: 提交**

```bash
git add tests/test_table.py src/deepeye_mcp/table.py
git commit -m "feat(table): add table JSON to Markdown conversion"
```

---

### Task 4: `extract_table` 工具

**Files:**
- Modify: `src/deepeye_mcp/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `_run_vision`（tools.py 内部）、`_extract_json`（tools.py 内部）、`json_to_markdown`/`has_merged_cells`/`_TABLE_JSON_PROMPT`（Task 3）、`classify_error`（Task 1）
- Produces: `extract_table(image_source: str, model: str | None = None) -> list[TextContent]`

- [ ] **Step 1: 写失败测试**（追加到 test_tools.py；导入 `extract_table`）

```python
@patch("deepeye_mcp.tools.create_vision_adapter")
async def test_extract_table_returns_markdown(mock_factory):
    mock_adapter = _build_mock_adapter(
        return_value='{"columns":2,"title":"t","rows":['
        '{"cells":[{"text":"n","rowspan":1,"colspan":1},{"text":"p"}]},'
        '{"cells":[{"text":"a","rowspan":1,"colspan":1},{"text":"1"}]}]}'
    )
    mock_factory.return_value = mock_adapter

    result = await extract_table(image_source=_DATA_URI)

    assert "| n | p |" in result[0].text
    assert "| a | 1 |" in result[0].text


@patch("deepeye_mcp.tools.create_vision_adapter")
async def test_extract_table_merged_appends_json(mock_factory):
    mock_adapter = _build_mock_adapter(
        return_value='{"columns":1,"title":"","rows":['
        '{"cells":[{"text":"A","rowspan":2,"colspan":1}]},'
        '{"cells":[{"text":"B","rowspan":1,"colspan":1}]}]}'
    )
    mock_factory.return_value = mock_adapter

    result = await extract_table(image_source=_DATA_URI)

    assert "```json" in result[0].text
    assert '"rowspan": 2' in result[0].text


@patch("deepeye_mcp.tools.create_vision_adapter")
async def test_extract_table_invalid_json_degraded(mock_factory):
    mock_adapter = _build_mock_adapter(return_value="无法识别")
    mock_factory.return_value = mock_adapter

    result = await extract_table(image_source=_DATA_URI)

    assert "表格提取失败" in result[0].text
```

注意：`_build_mock_adapter` 返回固定值，`extract_table` 内部重试时每次 `describe` 都返回同一值。改成 `side_effect` 时注意 `_run_vision` 的缓存关闭（`use_cache=False`）不会干扰。

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=src python -m pytest tests/test_tools.py -k extract_table -q`
Expected: FAIL，`ImportError`（`extract_table` 未定义）

- [ ] **Step 3: 实现**（tools.py）

顶部导入：
```python
from deepeye_mcp.table import _TABLE_JSON_PROMPT, has_merged_cells, json_to_markdown
```

新增常量与函数：
```python
_TABLE_RETRIES = 3


async def extract_table(
    image_source: str,
    model: str | None = None,
) -> list[TextContent]:
    """提取图片中的表格为 Markdown；复杂表格（合并单元格）附带 JSON 结构。

    Args:
        image_source: 图像来源（本地路径 / URL / data URI）。
        model: 可选模型名称覆盖。

    Returns:
        含 Markdown 表格的 ``list[TextContent]``。
    """
    try:
        for _ in range(_TABLE_RETRIES + 1):
            # 先尝试 json_object；后端不支持时降级为普通请求
            try:
                text = await _run_vision(
                    image_source,
                    _TABLE_JSON_PROMPT,
                    model,
                    use_cache=False,
                    max_tokens=8192,
                    reasoning_effort="low",
                    response_format={"type": "json_object"},
                )
            except httpx.HTTPStatusError:
                text = await _run_vision(
                    image_source,
                    _TABLE_JSON_PROMPT,
                    model,
                    use_cache=False,
                    max_tokens=8192,
                    reasoning_effort=None,
                    response_format=None,
                )
            json_str = _extract_json(text, require_key="rows")
            if json_str is not None:
                parsed = json.loads(json_str)
                md = json_to_markdown(parsed)
                if has_merged_cells(parsed):
                    return [TextContent(
                        type="text",
                        text=f"{md}\n\n（检测到合并单元格，附 JSON 完整结构）\n```json\n{json_str}\n```",
                    )]
                return [TextContent(type="text", text=md)]
        return [TextContent(type="text", text="表格提取失败：模型多次未返回有效表格 JSON")]
    except Exception as exc:
        return [TextContent(type="text", text=f"表格提取失败：{classify_error(exc, settings.vision_provider)[1]}")]
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=src python -m pytest tests/test_tools.py -k extract_table -q`
Expected: PASS（3 个测试）

- [ ] **Step 5: 提交**

```bash
git add src/deepeye_mcp/tools.py tests/test_tools.py
git commit -m "feat(tools): add extract_table tool"
```

---

### Task 5: 适配器新增 `describe_text`（纯文本请求）

**Files:**
- Modify: `src/deepeye_mcp/vision/base.py`、`openai_adapter.py`、`custom_adapter.py`、`gemini_adapter.py`
- Test: `tests/test_adapters.py`

**Interfaces:**
- Produces: `VisionAdapter.describe_text(self, prompt: str) -> str`（抽象方法，三个适配器实现）
- Consumes（Task 6 用）: `create_vision_adapter(model).describe_text(prompt)`

- [ ] **Step 1: 写失败测试**（追加到 test_adapters.py）

```python
@patch("deepeye_mcp.vision.custom_adapter.httpx.AsyncClient")
async def test_custom_describe_text_no_image(mock_client_cls):
    """describe_text 发送纯文本请求，payload 不含 image_url。"""
    payload = {"choices": [{"message": {"content": "summary"}}]}
    fake_client = _make_fake_client(payload)
    mock_client_cls.return_value = fake_client

    adapter = CustomVisionAdapter(
        model="qwen-vl-max",
        api_key="k",
        base_url="https://example.com/v1",
    )
    text = await adapter.describe_text("请总结")

    assert text == "summary"
    call = fake_client.post.await_args
    sent = call.kwargs["json"]
    assert sent["messages"][0]["content"] == "请总结"
    assert "image_url" not in str(sent["messages"])


@patch("deepeye_mcp.vision.gemini_adapter.httpx.AsyncClient")
async def test_gemini_describe_text(mock_client_cls):
    """Gemini describe_text 走 generateContent，parts 仅含 text。"""
    payload = {"candidates": [{"content": {"parts": [{"text": "summary"}]}}]}
    fake_client = _make_fake_client(payload)
    mock_client_cls.return_value = fake_client

    adapter = GeminiVisionAdapter(model="gemini-2.0-flash", api_key="k")
    text = await adapter.describe_text("请总结")

    assert text == "summary"
    call = fake_client.post.await_args
    sent = call.kwargs["json"]
    assert sent["contents"][0]["parts"][0]["text"] == "请总结"
    assert "inline_data" not in str(sent["contents"])
```

注意：`_make_fake_client` 生成的 fake 的 `post` 是 AsyncMock，`await_args` 可用。

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=src python -m pytest tests/test_adapters.py -k describe_text -q`
Expected: FAIL（`VisionAdapter` 无 `describe_text`）

- [ ] **Step 3: 实现**

base.py 抽象基类新增：
```python
    @abstractmethod
    async def describe_text(self, prompt: str) -> str:
        """纯文本请求（无图片），用于跨图汇总等场景。

        Args:
            prompt: 提示词。

        Returns:
            模型返回的文本。
        """
        ...
```

openai_adapter.py 新增（复用 `_client` 与重试模式）：
```python
    async def describe_text(self, prompt: str) -> str:
        """纯文本请求：messages 只有 user 文本，无图片。"""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": settings.max_tokens,
        }
        if settings.reasoning_effort:
            payload["reasoning_effort"] = settings.reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        client = _get_client()
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if not content:
            content = (message.get("reasoning_content") or "").strip()
        return content
```

custom_adapter.py 新增（每次新建 client，与现有 describe 一致）：
```python
    async def describe_text(self, prompt: str) -> str:
        """纯文本请求：messages 只有 user 文本，无图片。"""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": settings.max_tokens,
        }
        if settings.reasoning_effort:
            payload["reasoning_effort"] = settings.reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = settings.request_timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

        data = response.json()
        message = data["choices"][0]["message"]
        content = (message.get("content") or "").strip()
        if not content:
            content = (message.get("reasoning_content") or "").strip()
        return content
```

gemini_adapter.py 新增：
```python
    async def describe_text(self, prompt: str) -> str:
        """纯文本请求：generateContent parts 仅含 text，无 inline_data。"""
        url = f"{self.base_url.rstrip('/')}/models/{self.model}:generateContent"
        params = {"key": self.api_key}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        headers = {"Content-Type": "application/json"}
        timeout = settings.request_timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, params=params, json=payload, headers=headers)
            response.raise_for_status()

        data = response.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = None
        return (text or "").strip()
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=src python -m pytest tests/test_adapters.py -q`
Expected: PASS（全部）

- [ ] **Step 5: 提交**

```bash
git add src/deepeye_mcp/vision/ tests/test_adapters.py
git commit -m "feat(adapters): add describe_text for text-only requests"
```

---

### Task 6: `analyze_images` 批量工具

**Files:**
- Modify: `src/deepeye_mcp/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `_run_vision`、`classify_error`（Task 1）、`create_vision_adapter(model).describe_text`（Task 5）
- Produces: `analyze_images(image_sources: list[str], prompt: str = _DEFAULT_DESCRIBE_PROMPT, model: str | None = None) -> list[TextContent]`
- 内部辅助：`_source_name(source: str, index: int) -> str`

- [ ] **Step 1: 写失败测试**（追加到 test_tools.py；导入 `analyze_images`、`asyncio`）

```python
@patch("deepeye_mcp.tools.create_vision_adapter")
async def test_analyze_images_two_images(mock_factory):
    adapter = _build_mock_adapter(return_value="图像描述A")
    adapter.describe_text = AsyncMock(return_value="对比总结")
    mock_factory.return_value = adapter

    result = await analyze_images(image_sources=["a.png", "b.png"])

    text = result[0].text
    assert "[1] a.png: 图像描述A" in text
    assert "[2] b.png: 图像描述A" in text
    assert "【跨图汇总】" in text
    assert "对比总结" in text
    # 两条逐图 + 一条汇总，共 3 次模型调用
    assert adapter.describe.await_count == 2
    adapter.describe_text.assert_awaited_once()


@patch("deepeye_mcp.tools.create_vision_adapter")
async def test_analyze_images_isolated_failure(mock_factory):
    """单张失败不阻塞整体，返回分类后的占位文本。"""
    adapter = MagicMock()
    adapter.describe = AsyncMock(side_effect=[RuntimeError("boom"), "ok"])
    adapter.describe_text = AsyncMock(return_value="汇总")
    mock_factory.return_value = adapter

    result = await analyze_images(image_sources=["bad.png", "good.png"])

    text = result[0].text
    assert "boom" in text  # unknown 分类保留原始异常
    assert "[2] good.png: ok" in text
    assert "汇总" in text


@patch("deepeye_mcp.tools.create_vision_adapter")
async def test_analyze_images_empty_list(mock_factory):
    mock_factory.return_value = _build_mock_adapter()

    result = await analyze_images(image_sources=[])

    assert "不能为空" in result[0].text
    mock_factory.assert_not_called()
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=src python -m pytest tests/test_tools.py -k analyze_images -q`
Expected: FAIL（`analyze_images` 未定义）

- [ ] **Step 3: 实现**（tools.py）

顶部新增导入：
```python
import asyncio
from pathlib import Path

from deepeye_mcp.vision import create_vision_adapter
```
（`create_vision_adapter` tools.py 已导入，核对后勿重复）

新增函数：
```python
def _source_name(source: str, index: int) -> str:
    """从 image_source 提取展示名：本地路径取文件名，URL 取末段，data URI 用序号。"""
    if source.startswith(("http://", "https://")):
        return source.rstrip("/").split("/")[-1] or f"图片{index}"
    if source.startswith("data:"):
        return f"图片{index}"
    return Path(source).name


async def analyze_images(
    image_sources: list[str],
    prompt: str = _DEFAULT_DESCRIBE_PROMPT,
    model: str | None = None,
) -> list[TextContent]:
    """批量分析多张图片：逐图结果 + 跨图对比汇总。

    Args:
        image_sources: 多张图片来源（本地路径 / URL / data URI）。
        prompt: 应用于每张图的统一提示词。
        model: 可选模型名称覆盖。

    Returns:
        含逐图结果与汇总的 ``list[TextContent]``。
    """
    if not image_sources:
        return [TextContent(type="text", text="错误：image_sources 数组不能为空")]

    provider = settings.vision_provider
    try:
        per_image: list[str | BaseException] = await asyncio.gather(
            *[_run_vision(src, prompt, model) for src in image_sources],
            return_exceptions=True,
        )
    except Exception as exc:
        return [TextContent(type="text", text=f"批量分析失败：{classify_error(exc, provider)[1]}")]

    lines: list[str] = []
    for i, (src, result) in enumerate(zip(image_sources, per_image), 1):
        name = _source_name(src, i)
        if isinstance(result, BaseException):
            text = classify_error(result, provider)[1]
        else:
            text = result
        lines.append(f"[{i}] {name}: {text}")

    per_image_text = "\n\n".join(lines)

    # 汇总：逐图描述作为纯文本发给模型对比（不依赖多图支持）
    summary_prompt = (
        f"以下是同一 prompt 对多张图片的分析结果，请对比这些图片，"
        f"总结彼此的异同点和关键结论。\n\n{per_image_text}"
    )
    try:
        adapter = create_vision_adapter(model)
        summary = await adapter.describe_text(summary_prompt)
    except Exception as exc:
        summary = f"（汇总失败：{classify_error(exc, provider)[1]}）"

    return [TextContent(type="text", text=f"{per_image_text}\n\n【跨图汇总】\n{summary}")]
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=src python -m pytest tests/test_tools.py -q`
Expected: PASS（全部）

- [ ] **Step 5: 提交**

```bash
git add src/deepeye_mcp/tools.py tests/test_tools.py
git commit -m "feat(tools): add analyze_images bulk tool"
```

---

### Task 7: server.py 注册两个新工具

**Files:**
- Modify: `src/deepeye_mcp/server.py`

**Interfaces:**
- Consumes: `extract_table`、`analyze_images`（Task 4/6）

- [ ] **Step 1: 写失败测试**（追加到 test_tools.py）

```python
def test_server_tools_registered():
    """server 应注册 extract_table 与 analyze_images。"""
    from deepeye_mcp.server import _TOOLS

    names = {t.name for t in _TOOLS}
    assert "extract_table" in names
    assert "analyze_images" in names
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=src python -m pytest tests/test_tools.py -k server_tools -q`
Expected: FAIL（`extract_table`/`analyze_images` 不在 `_TOOLS`）

- [ ] **Step 3: 实现**（server.py）

导入：
```python
from deepeye_mcp.tools import (
    analyze_images,
    analyze_layout,
    ask_about_image,
    describe_image,
    extract_table,
    extract_text,
)
```

`_TOOLS` 追加两个 Tool 定义：
```python
    Tool(
        name="extract_table",
        description="提取图片中的表格为 Markdown（截图/纸质/图表/合并单元格均支持）；复杂表格附带 JSON 结构。",
        inputSchema={
            "type": "object",
            "properties": {
                "image_source": {"type": "string", "description": "图像来源：本地路径、http(s) URL 或 data URI。"},
                "model": {"type": "string", "description": "可选模型名称覆盖。"},
            },
            "required": ["image_source"],
        },
    ),
    Tool(
        name="analyze_images",
        description="批量分析多张图片：逐图返回结果 + 跨图对比汇总。",
        inputSchema={
            "type": "object",
            "properties": {
                "image_sources": {"type": "array", "items": {"type": "string"}, "description": "多张图片来源（本地路径 / URL / data URI）。"},
                "prompt": {"type": "string", "description": "应用于每张图的统一提示词，可选。"},
                "model": {"type": "string", "description": "可选模型名称覆盖。"},
            },
            "required": ["image_sources"],
        },
    ),
```

`call_tool` 分发新增：
```python
    elif name == "extract_table":
        content = await extract_table(
            image_source=arguments["image_source"],
            model=arguments.get("model"),
        )
    elif name == "analyze_images":
        content = await analyze_images(
            image_sources=arguments["image_sources"],
            prompt=arguments.get("prompt"),
            model=arguments.get("model"),
        )
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONPATH=src python -m pytest tests/test_tools.py -k server_tools -q`
Expected: PASS

随后全量回归：
Run: `PYTHONPATH=src python -m pytest tests/ -q --basetemp=.pytest_tmp -p no:cacheprovider`
Expected: PASS（全部）

- [ ] **Step 5: 提交**

```bash
git add src/deepeye_mcp/server.py tests/test_tools.py
git commit -m "feat(server): register extract_table and analyze_images"
```

---

### Task 8: README 更新

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 在「工具一览」追加两个工具说明**

在 `analyze_layout` 小节之后追加：

```markdown
### `extract_table` — 图片转表格

提取图片中的表格为 Markdown（截图数据表 / 纸质表格 / 图表 / 合并单元格均支持）。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `image_source` | string | 是 | 本地路径 / http(s) URL / data URI |
| `model` | string | 否 | 临时指定视觉模型 |

**返回**：Markdown 表格；检测到合并单元格时追加 JSON 结构（含 rowspan/colspan）。

### `analyze_images` — 批量图片分析

对多张图片应用同一提示词，返回逐图结果 + 跨图对比汇总。

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `image_sources` | string[] | 是 | 多张图片来源（本地路径 / URL / data URI） |
| `prompt` | string | 否 | 每张图统一提示词，默认详细描述 |
| `model` | string | 否 | 临时指定视觉模型 |

**返回**：`[序号] 来源: 结果` 列表 + `【跨图汇总】` 对比总结。
```

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "docs(readme): add extract_table and analyze_images tools"
```

---

## Self-Review

**Spec 覆盖检查：**
- `extract_table`（含合并单元格 JSON 追加、降级容错）→ Task 3 + Task 4 ✓
- `analyze_images`（数组入参、并发、逐图+汇总、失败隔离）→ Task 5 + Task 6 ✓
- 错误分类（6 类、6 工具统一）→ Task 1 + Task 2 ✓
- server 注册 → Task 7 ✓
- README → Task 8 ✓

**占位符：** 无 TBD/TODO，所有步骤含真实代码。

**类型一致性：**
- `classify_error(exc, provider)` 返回 `(str, str)`：Task 1 定义，Task 2/4/6 使用 ✓
- `describe_text(prompt) -> str`：Task 5 定义，Task 6 使用 ✓
- `json_to_markdown(parsed) -> str`、`has_merged_cells(parsed) -> bool`：Task 3 定义，Task 4 使用 ✓
- `extract_table(image_source, model)`、`analyze_images(image_sources, prompt, model)` → `list[TextContent]`：Task 4/6 定义，Task 7 使用 ✓