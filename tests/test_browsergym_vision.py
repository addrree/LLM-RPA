import base64
import json
import sys
import types

from app.browsergym_integration.report import BrowserGymRunReport, BrowserGymStepRecord
from app.browsergym_integration.vision import _image_like_to_png_base64, extract_browsergym_image_base64


class _FakeArray:
    shape = (2, 2, 3)
    dtype = "uint8"

    def __bool__(self):
        raise ValueError("truth value of an array is ambiguous")

    def tobytes(self):
        return bytes([
            255, 0, 0,
            0, 255, 0,
            0, 0, 255,
            255, 255, 255,
        ])


def _assert_png_b64(value: str):
    raw = base64.b64decode(value)
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")


def test_array_like_uint8_hwc_to_base64_png():
    encoded = extract_browsergym_image_base64({"screenshot": _FakeArray()}, None)
    assert isinstance(encoded, str)
    _assert_png_b64(encoded)


def test_array_bool_raising_value_error_does_not_break():
    encoded = extract_browsergym_image_base64({"image": _FakeArray()}, {})
    assert isinstance(encoded, str)
    _assert_png_b64(encoded)


def test_none_returns_none():
    assert extract_browsergym_image_base64({"screenshot": None}, {"image": None}) is None
    assert _image_like_to_png_base64(None) is None


def test_pil_image_to_base64_string_with_fake_pil(monkeypatch):
    class FakeImageType:
        mode = "RGB"

        def save(self, buf, format):
            assert format == "PNG"
            buf.write(b"\x89PNG\r\n\x1a\nfake")

    fake_image_module = types.SimpleNamespace(Image=FakeImageType)
    monkeypatch.setitem(sys.modules, "PIL", types.SimpleNamespace(Image=fake_image_module))
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image_module)

    encoded = _image_like_to_png_base64(FakeImageType())
    assert isinstance(encoded, str)
    assert base64.b64decode(encoded).startswith(b"\x89PNG")


def test_base64_not_serialized_in_report_context():
    encoded = extract_browsergym_image_base64({"screenshot": _FakeArray()}, None)
    report = BrowserGymRunReport(
        env_id="browsergym/openended",
        goal="g",
        status="partial",
        steps=[
            BrowserGymStepRecord(
                step_idx=0,
                action="noop()",
                vision_used=True,
                vision_image_present=True,
                internal_plan={"goal": "g"},
                selected_step={"action": "noop", "args": {}},
            )
        ],
    )
    payload = json.dumps(report.model_dump(mode="json"))
    assert encoded not in payload
    assert "screenshot" not in payload.lower()
