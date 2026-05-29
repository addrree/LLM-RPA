from __future__ import annotations

import base64
import binascii
import io
import struct
import zlib
from typing import Any


def extract_browsergym_image_base64(obs: dict, info: dict | None = None) -> str | None:
    """Extract a BrowserGym screenshot/image as PNG base64 without serializing raw arrays."""
    candidates: list[Any] = []
    if isinstance(obs, dict):
        for key in ("screenshot", "image", "screenshot_base64", "image_base64"):
            if key in obs:
                candidates.append(obs.get(key))
    if isinstance(info, dict):
        for key in ("screenshot", "image", "screenshot_base64", "image_base64"):
            if key in info:
                candidates.append(info.get(key))

    for candidate in candidates:
        if candidate is not None:
            encoded = _image_like_to_png_base64(candidate)
            if encoded is not None:
                return encoded
    return None


def _image_like_to_png_base64(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("data:image") and "," in stripped:
            stripped = stripped.split(",", 1)[1]
        try:
            base64.b64decode(stripped, validate=True)
        except (binascii.Error, ValueError):
            return None
        return stripped
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        if _looks_like_image_bytes(raw):
            return base64.b64encode(raw).decode("ascii")
        return None

    pil_encoded = _pil_image_to_png_base64(value)
    if pil_encoded is not None:
        return pil_encoded

    return _array_like_to_png_base64(value)


def _looks_like_image_bytes(raw: bytes) -> bool:
    return raw.startswith(b"\x89PNG\r\n\x1a\n") or raw.startswith(b"\xff\xd8\xff") or raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a") or raw.startswith(b"RIFF")


def _pil_image_to_png_base64(value: Any) -> str | None:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    if not isinstance(value, Image.Image):
        return None
    try:
        img = value.convert("RGBA") if value.mode not in {"RGB", "RGBA", "L"} else value
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _array_like_to_png_base64(value: Any) -> str | None:
    shape = getattr(value, "shape", None)
    if not isinstance(shape, tuple):
        return None
    if len(shape) not in {2, 3}:
        return None
    if len(shape) == 2:
        height, width = shape
        channels = 1
    else:
        height, width, channels = shape
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        return None
    if channels not in {1, 3, 4}:
        return None

    try:
        raw, mode = _coerce_array_bytes(value, height=height, width=width, channels=channels)
    except Exception:
        return None
    if raw is None or mode is None:
        return None

    # Prefer Pillow when available, but keep a tiny PNG fallback so tests and
    # sidecar diagnostics remain usable in minimal environments.
    try:
        from PIL import Image  # type: ignore

        img = Image.frombytes(mode, (width, height), raw)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        png = _encode_png(raw, width=width, height=height, mode=mode)
        return base64.b64encode(png).decode("ascii") if png is not None else None


def _coerce_array_bytes(value: Any, *, height: int, width: int, channels: int) -> tuple[bytes | None, str | None]:
    mode = "L" if channels == 1 else "RGB" if channels == 3 else "RGBA"
    dtype = str(getattr(value, "dtype", "")).lower()

    try:
        import numpy as np  # type: ignore

        arr = np.asarray(value)
        if arr.ndim == 2:
            pass
        elif arr.ndim == 3 and arr.shape[2] in {1, 3, 4}:
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
                mode = "L"
            elif arr.shape[2] == 4:
                mode = "RGBA"
            else:
                mode = "RGB"
        else:
            raise ValueError("unsupported numpy array shape")
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
            arr = arr.astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if _declares_bgr(value) and arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr.copy()
            arr[..., :3] = arr[..., 2::-1]
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        return arr.tobytes(), mode
    except Exception:
        pass

    if "float" in dtype:
        data = _flatten_nested(value)
        if data is None:
            return None, None
        raw = bytes(max(0, min(255, int(round(float(x) * 255)))) for x in data)
        return raw, mode

    if hasattr(value, "tobytes"):
        raw = value.tobytes()
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        data = _flatten_nested(value)
        if data is None:
            return None, None
        raw = bytes(max(0, min(255, int(x))) for x in data)
    expected = height * width * channels
    if len(raw) != expected:
        return None, None
    if _declares_bgr(value) and channels in {3, 4}:
        raw = _swap_bgr_to_rgb(raw, channels=channels)
    return raw, mode


def _declares_bgr(value: Any) -> bool:
    for attr in ("channel_order", "color_order", "mode", "format"):
        marker = getattr(value, attr, None)
        if isinstance(marker, str) and marker.upper() == "BGR":
            return True
    return False


def _swap_bgr_to_rgb(raw: bytes, *, channels: int) -> bytes:
    data = bytearray(raw)
    for idx in range(0, len(data), channels):
        data[idx], data[idx + 2] = data[idx + 2], data[idx]
    return bytes(data)


def _flatten_nested(value: Any) -> list[Any] | None:
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            return None
    flat: list[Any] = []

    def walk(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
        else:
            flat.append(item)

    try:
        walk(value)
    except Exception:
        return None
    return flat


def _encode_png(raw: bytes, *, width: int, height: int, mode: str) -> bytes | None:
    color_type_by_mode = {"L": 0, "RGB": 2, "RGBA": 6}
    channels_by_mode = {"L": 1, "RGB": 3, "RGBA": 4}
    color_type = color_type_by_mode.get(mode)
    channels = channels_by_mode.get(mode)
    if color_type is None or channels is None:
        return None
    stride = width * channels
    if len(raw) != stride * height:
        return None

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    scanlines = b"".join(b"\x00" + raw[y * stride : (y + 1) * stride] for y in range(height))
    ihdr = struct.pack("!IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b"")
