"""The LDM application exposes only litter-detection, CCTV and login routes.

Invoice extraction was moved out of this project; these checks keep it from being re-registered by accident.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import create_app


def test_no_invoice_routes_in_openapi_schema(cfg):
    schema = create_app(cfg).openapi()
    assert not [p for p in schema["paths"] if "invoice" in p.lower()]
    tags = {t for ops in schema["paths"].values() for op in ops.values() for t in op.get("tags", [])}
    assert "invoices" not in tags


def test_invoice_endpoints_are_not_served(cfg):
    with TestClient(create_app(cfg)) as c:
        assert c.get("/api/invoices/engine").status_code == 404
        assert c.post("/api/invoices/extract").status_code == 404
        assert c.get("/api/health").status_code == 200
