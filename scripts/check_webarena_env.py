#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys

import requests

from app.browsergym_integration.config import WEBARENA_REQUIRED_ENV_VARS


def check_url(url: str) -> bool:
    try:
        r = requests.head(url, timeout=5, allow_redirects=True)
        if r.status_code >= 400:
            r = requests.get(url, timeout=5)
        return r.status_code < 400
    except Exception:
        return False


def main() -> int:
    package_versions = {}
    imports_ok = True
    for m in ["browsergym", "browsergym.webarena", "gymnasium"]:
        try:
            mod = importlib.import_module(m)
            package_versions[m] = getattr(mod, "__version__", "unknown")
        except Exception as exc:
            imports_ok = False
            package_versions[m] = f"missing: {exc}"

    missing_env = [k for k in WEBARENA_REQUIRED_ENV_VARS if not os.getenv(k)]
    urls = {k: os.getenv(k) for k in WEBARENA_REQUIRED_ENV_VARS if os.getenv(k)}
    unreachable = [k for k, v in urls.items() if not check_url(v)]

    pw = subprocess.run([sys.executable, "-m", "playwright", "install", "--dry-run"], capture_output=True, text=True)
    playwright_ok = pw.returncode == 0

    result = {
        "ok": imports_ok and playwright_ok and not missing_env and not unreachable,
        "missing_env": missing_env,
        "unreachable_urls": unreachable,
        "package_versions": package_versions,
        "playwright_browsers_ok": playwright_ok,
        "python_executable": sys.executable,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
