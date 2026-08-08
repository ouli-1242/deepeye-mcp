"""图像源解析工具。

统一处理三种图像来源：
- 本地文件路径
- 公网 URL（http/https）
- Base64 data URI（``data:image/...;base64,...``）

所有解析函数最终返回 ``(base64_data, mime_type)`` 元组。
"""

from __future__ import annotations

import base64
import io
import ipaddress
import mimetypes
import socket
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image

from deepeye_mcp.config import settings

_DEFAULT_MIME = "image/png"


def _is_public_ip(ip: str) -> bool:
    """判断 IP 是否为公网地址。

    内网 / 回环 / 链路本地 / 保留 / 组播 / 未指定地址均视为非公网。
    """
    try:
        addr = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _ensure_public_target(url: str) -> str:
    """校验 URL 指向公网地址，防止 SSRF。

    解析主机名到 IP，任一解析结果落在内网/保留地址即拒绝；
    ``settings.allow_private_urls=True`` 时跳过校验（仅本地调试）。

    Raises:
        ValueError: URL 无法解析或解析到非公网地址时抛出。
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL 缺少主机名")
    if settings.allow_private_urls:
        return hostname

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"无法解析主机: {hostname}") from exc

    for info in infos:
        ip = info[4][0]
        if not _is_public_ip(ip):
            raise ValueError(
                f"拒绝访问非公网地址（SSRF 防护）: {hostname} ({ip})"
            )
    return hostname


def load_image_as_base64(path: str) -> tuple[str, str]:
    """同步读取本地图片文件并返回 ``(base64_data, mime_type)``。

    Args:
        path: 本地图片文件路径。

    Returns:
        ``(base64_data, mime_type)`` 元组。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"图像文件不存在: {path}")

    data = p.read_bytes()
    b64_data = base64.b64encode(data).decode("ascii")
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = _DEFAULT_MIME
    return b64_data, mime_type


async def load_image_from_url_as_base64(url: str) -> tuple[str, str]:
    """异步下载 URL 图片并返回 ``(base64_data, mime_type)``。

    Args:
        url: 以 ``http://`` 或 ``https://`` 开头的图片地址。

    Returns:
        ``(base64_data, mime_type)`` 元组。MIME 从响应 ``Content-Type``
        推断（取 ``;`` 之前部分），无法推断时默认 ``image/png``。
    """
    async with httpx.AsyncClient(
        follow_redirects=True, timeout=settings.request_timeout
    ) as client:
        _ensure_public_target(url)
        response = await client.get(url)
        response.raise_for_status()

    data = response.content
    if len(data) > settings.max_image_bytes:
        raise ValueError(
            f"图片超过大小上限: {len(data)} > {settings.max_image_bytes} bytes"
        )
    b64_data = base64.b64encode(data).decode("ascii")
    content_type = response.headers.get("Content-Type", "")
    mime_type = content_type.split(";")[0].strip() if content_type else ""
    if not mime_type:
        mime_type = _DEFAULT_MIME
    return b64_data, mime_type


def _parse_data_uri(image_source: str) -> tuple[str, str]:
    """解析 ``data:image/...;base64,...`` 形式的 data URI。

    Args:
        image_source: data URI 字符串。

    Returns:
        ``(base64_data, mime_type)`` 元组。
    """
    # 形如: data:image/jpeg;base64,/9j/...
    header, _, b64_data = image_source.partition(",")
    # header 形如: data:image/jpeg;base64
    mime_type = _DEFAULT_MIME
    if header.startswith("data:"):
        meta = header[len("data:") :]  # image/jpeg;base64
        if ";" in meta:
            mime_type = meta.split(";")[0].strip() or _DEFAULT_MIME
        elif meta:
            mime_type = meta.strip() or _DEFAULT_MIME
    if not b64_data:
        raise ValueError("data URI 不含 base64 数据")
    return b64_data, mime_type


async def parse_image_source(image_source: str) -> tuple[str, str]:
    """统一图像源解析入口。

    根据前缀自动选择解析策略：
    - ``data:`` 开头 → 直接解析 data URI，不进行 IO
    - ``http://`` / ``https://`` 开头 → 异步下载
    - 其他 → 视为本地路径

    Args:
        image_source: 图像来源字符串。

    Returns:
        ``(base64_data, mime_type)`` 元组。
    """
    if image_source.startswith("data:"):
        return _parse_data_uri(image_source)
    if image_source.startswith(("http://", "https://")):
        return await load_image_from_url_as_base64(image_source)
    return load_image_as_base64(image_source)


def preprocess_image(b64: str, mime: str) -> tuple[str, str]:
    """图片预处理：超过最大边长时等比缩放并转 JPEG 以减小体积。

    处理规则：
    - ``settings.image_max_dim == 0`` 时禁用预处理，原样返回 ``(b64, mime)``。
    - 图片最大边未超过 ``image_max_dim`` 时原样返回。
    - 超过则等比缩放到 ``image_max_dim``，再以 JPEG 重新编码，
      返回的 ``mime`` 改为 ``image/jpeg``。
    - Pillow 解码/处理任何异常时原样返回（不阻断主流程，保证健壮性）。

    Args:
        b64: 原始图片的 base64 字符串。
        mime: 原始图片的 MIME 类型。

    Returns:
        ``(new_b64, new_mime)`` 元组。
    """
    max_dim = settings.image_max_dim
    if not max_dim or max_dim <= 0:
        return b64, mime

    try:
        raw = base64.b64decode(b64)
        with Image.open(io.BytesIO(raw)) as img:
            width, height = img.size
            longest = max(width, height)
            if longest <= max_dim:
                return b64, mime

            # 等比缩放
            scale = max_dim / longest
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            img = img.convert("RGB")
            img = img.resize(new_size, Image.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            new_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            return new_b64, "image/jpeg"
    except Exception:
        # 任何解码/处理异常均原样返回，不阻断主流程
        return b64, mime
