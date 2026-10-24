"""Tests for the HTTP API.

Uses FastAPI's TestClient, so no server needs to be running.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from sagex.web.app import app

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

PASSPHRASE = "web test passphrase"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _png_bytes(width: int, height: int, *, flat: bool = False) -> bytes:
    """A PNG upload fixture. Textured by default, flat on request."""
    if flat:
        array = np.full((height, width, 3), 128, dtype=np.uint8)
    else:
        array = np.random.default_rng(99).integers(
            30, 225, size=(height, width, 3), dtype=np.uint8
        )
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def cover_png() -> bytes:
    return _png_bytes(256, 256)


# ------------------------------------------------------------------ basics

def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "SAGE-X" in response.text


# ---------------------------------------------------------------- capacity

def test_capacity_reports_every_rate(client, cover_png):
    response = client.post("/api/capacity", files={"image": ("c.png", cover_png, "image/png")})
    assert response.status_code == 200

    body = response.json()
    assert (body["width"], body["height"]) == (256, 256)
    assert set(body["rates"]) == {"conservative", "balanced", "aggressive"}
    assert (body["rates"]["aggressive"]["max_payload_bytes"]
            > body["rates"]["conservative"]["max_payload_bytes"])


def test_capacity_rejects_non_image(client):
    response = client.post("/api/capacity",
                           files={"image": ("x.png", b"not an image", "image/png")})
    assert response.status_code == 400


def test_capacity_rejects_empty_upload(client):
    response = client.post("/api/capacity", files={"image": ("x.png", b"", "image/png")})
    assert response.status_code == 400


# ------------------------------------------------------------- round trip

def test_hide_then_reveal_returns_original(client, cover_png):
    secret = b"the quick brown fox jumps over the lazy dog" * 8

    hidden = client.post(
        "/api/hide",
        files={"cover": ("c.png", cover_png, "image/png"),
               "payload": ("s.txt", secret, "text/plain")},
        data={"passphrase": PASSPHRASE, "rate": "balanced"},
    )
    assert hidden.status_code == 200
    assert hidden.headers["content-type"] == "image/png"
    assert int(hidden.headers["x-sagex-payload-bytes"]) == len(secret)
    assert float(hidden.headers["x-sagex-psnr"]) > 50

    revealed = client.post(
        "/api/reveal",
        files={"stego": ("carrier.png", hidden.content, "image/png")},
        data={"passphrase": PASSPHRASE, "filename": "out.txt"},
    )
    assert revealed.status_code == 200
    assert revealed.content == secret
    assert "out.txt" in revealed.headers["content-disposition"]


# --------------------------------------------------------------- refusals

def test_oversized_payload_is_refused_with_explanation(client):
    """The behaviour that distinguishes this tool: refuse, and say why."""
    small = _png_bytes(64, 64)
    response = client.post(
        "/api/hide",
        files={"cover": ("c.png", small, "image/png"),
               "payload": ("s.bin", b"x" * 100_000, "application/octet-stream")},
        data={"passphrase": PASSPHRASE, "rate": "conservative"},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "holds at most" in detail


def test_wrong_passphrase_is_rejected(client, cover_png):
    hidden = client.post(
        "/api/hide",
        files={"cover": ("c.png", cover_png, "image/png"),
               "payload": ("s.txt", b"classified", "text/plain")},
        data={"passphrase": PASSPHRASE, "rate": "balanced"},
    )
    response = client.post(
        "/api/reveal",
        files={"stego": ("carrier.png", hidden.content, "image/png")},
        data={"passphrase": "not the passphrase"},
    )
    assert response.status_code in (403, 404)


def test_image_with_no_message_reports_not_found(client, cover_png):
    response = client.post(
        "/api/reveal",
        files={"stego": ("plain.png", cover_png, "image/png")},
        data={"passphrase": PASSPHRASE},
    )
    assert response.status_code == 404


def test_empty_passphrase_is_rejected(client, cover_png):
    """An empty passphrase must never succeed.

    Either status is a correct rejection: the HTTP client omits empty form fields
    altogether, so FastAPI reports a missing field (422). A browser sends the field
    present-but-empty, which reaches the handler's own check (400).
    """
    response = client.post(
        "/api/hide",
        files={"cover": ("c.png", cover_png, "image/png"),
               "payload": ("s.txt", b"data", "text/plain")},
        data={"passphrase": "", "rate": "balanced"},
    )
    assert response.status_code in (400, 422)


def test_unknown_rate_is_rejected(client, cover_png):
    response = client.post(
        "/api/hide",
        files={"cover": ("c.png", cover_png, "image/png"),
               "payload": ("s.txt", b"data", "text/plain")},
        data={"passphrase": PASSPHRASE, "rate": "reckless"},
    )
    assert response.status_code == 400


# ------------------------------------------------------------------- rank

def test_rank_orders_by_texture(client):
    """A flat image is a worse cover than a textured one of the same size."""
    response = client.post(
        "/api/rank",
        files=[("images", ("flat.png", _png_bytes(256, 256, flat=True), "image/png")),
               ("images", ("busy.png", _png_bytes(256, 256), "image/png"))],
        data={"size": "1000", "rate": "balanced"},
    )
    assert response.status_code == 200

    ranked = response.json()["ranked"]
    assert [entry["filename"] for entry in ranked][0] == "busy.png"
    assert ranked[0]["texture_score"] > ranked[-1]["texture_score"]


def test_rank_separates_images_that_cannot_fit(client):
    # A 256x256 holds ~2.4 KB at the balanced rate, a 32x32 holds nothing at all
    # once the 54 bytes of overhead are counted.
    response = client.post(
        "/api/rank",
        files=[("images", ("tiny.png", _png_bytes(32, 32), "image/png")),
               ("images", ("big.png", _png_bytes(256, 256), "image/png"))],
        data={"size": "1000", "rate": "balanced"},
    )
    body = response.json()
    assert [entry["filename"] for entry in body["ranked"]] == ["big.png"]
    assert [entry["filename"] for entry in body["rejected"]] == ["tiny.png"]
