# -*- coding: utf-8 -*-
"""pytest 公共配置：路径注入 + 服务可用性检测。

集成测试（test_api_smoke.py）需要 8011 服务在跑；不在就自动 skip，
保证"单元测试永远可跑、集成测试有服务才跑"。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))   # 让测试能 import sf6_rag.*

API = os.environ.get("W2_API_BASE", "http://127.0.0.1:8011")


def _alive() -> bool:
    try:
        urllib.request.urlopen(API + "/health", timeout=3)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def api_base() -> str:
    if not _alive():
        pytest.skip(f"服务未启动（{API}），跳过集成测试")
    return API


@pytest.fixture(scope="session")
def token(api_base: str) -> str:
    """登录拿 token（服务进程内存 token）。"""
    req = urllib.request.Request(
        api_base + "/login",
        data=json.dumps({"username": "admin", "password": "admin123"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())["token"]
