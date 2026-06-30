# tests/test_api.py

import pytest
import io
import sys; sys.path.insert(0, 'src')
from fastapi.testclient import TestClient


def get_test_client():
    """延迟导入, 避免测试收集阶段加载torch"""
    from api import app
    return TestClient(app)


def test_health():
    client = get_test_client()
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "gpu_available" in data


def test_measure_missing_file():
    client = get_test_client()
    resp = client.post("/api/measure", files={}, data={"height_cm": 170})
    assert resp.status_code == 422  # FastAPI validation error


def test_measure_invalid_height():
    client = get_test_client()
    fake_img = io.BytesIO(b"\x00" * 600)
    resp = client.post(
        "/api/measure",
        files={"front": ("f.jpg", fake_img, "image/jpeg"), "side": ("s.jpg", fake_img, "image/jpeg")},
        data={"height_cm": "not_a_number"}
    )
    assert resp.status_code == 422


def test_root_returns_html():
    client = get_test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
