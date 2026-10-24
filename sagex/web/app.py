"""HTTP API and web interface.

Run with::

    sagex-web
    # or: uvicorn sagex.web.app:app --reload

Then open http://127.0.0.1:3000

Concurrency:
    The route handlers are deliberately plain ``def``, not ``async def``. All the real
    work here is synchronous and CPU-bound (NumPy array manipulation and Argon2id key
    derivation), and blocking work inside an ``async def`` handler runs *on* the event
    loop, stalling every other request until it finishes. FastAPI runs plain ``def``
    handlers in a threadpool instead. Both NumPy and Argon2 release the GIL while
    working, so that gives real parallelism rather than just interleaving.

Scope note:
    This is intended to run on localhost for demonstration. Passphrases travel in the
    request body, so exposing it on a network without TLS would leak them, and there
    is no authentication.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .. import capacity, imageio
from ..core.crypto import DecryptionError
from ..core.frame import FrameError
from ..embed import costs
from ..pipeline import hide, reveal

#: Refuse anything larger. The cost here is decoding and re-encoding the PNG, plus
#: holding the pixel array, so a 12-megapixel cover is already a few seconds per call.
MAX_UPLOAD_BYTES = 40 * 1024 * 1024

RATES = {
    "conservative": capacity.RATE_CONSERVATIVE,
    "balanced": capacity.RATE_BALANCED,
    "aggressive": capacity.RATE_AGGRESSIVE,
}

#: Port the ``sagex-web`` command serves on. start.sh uses the same default, and
#: accepts an override as its first argument.
DEFAULT_PORT = 3000

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="SAGE-X", description="Hide encrypted data inside images.")


# ----------------------------------------------------------------- helpers

def _read_upload(upload: UploadFile) -> bytes:
    data = upload.file.read()
    if not data:
        raise HTTPException(400, f"{upload.filename or 'file'} is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"{upload.filename} is {len(data) / 2**20:.1f} MB; the limit is "
            f"{MAX_UPLOAD_BYTES // 2**20} MB",
        )
    return data


def _decode_image(data: bytes, label: str) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as handle:
            return np.asarray(handle.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, f"{label} is not a readable image") from exc


def _resolve_rate(name: str) -> float:
    if name not in RATES:
        raise HTTPException(400, f"unknown rate {name!r}; expected one of {list(RATES)}")
    return RATES[name]


def _report_json(report: capacity.Capacity) -> dict:
    return {
        "rate": report.rate,
        "rate_name": report.rate_name,
        "description": report.description,
        "max_payload_bytes": report.max_payload_bytes,
        # None when texture scoring was skipped, so guard before rounding.
        "texture_score": (None if report.texture_score is None
                          else round(report.texture_score, 3)),
        "usable": report.is_usable,
        "summary": report.summary(),
    }


# ----------------------------------------------------------------- routes

@app.post("/api/capacity")
def api_capacity(image: UploadFile = File(...)):
    """Report what an image can hold, at every rate.

    Called as soon as a cover is chosen, so the interface can tell the user what
    fits before they pick a file to hide.
    """
    array = _decode_image(_read_upload(image), image.filename or "image")
    # Texture is a property of the image, not the rate. Scoring it costs a full HILL
    # pass, so do it once and reuse it across all three reports.
    texture = costs.texture_score(imageio.to_luminance(array))
    return {
        "filename": image.filename,
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "rates": {name: _report_json(capacity.analyse(array, rate, texture=texture))
                  for name, rate in RATES.items()},
    }


@app.post("/api/hide")
def api_hide(
    cover: UploadFile = File(...),
    payload: UploadFile = File(...),
    passphrase: str = Form(...),
    rate: str = Form("balanced"),
):
    """Hide a file inside an image and return the stego PNG."""
    if not passphrase:
        raise HTTPException(400, "passphrase must not be empty")

    cover_array = _decode_image(_read_upload(cover), cover.filename or "cover")
    secret = _read_upload(payload)

    try:
        result = hide(cover_array, secret, passphrase, rate=_resolve_rate(rate))
    except capacity.InsufficientCapacity as exc:
        # 422: the request was well-formed, the data just does not fit. This is the
        # refusal that distinguishes the tool, so the message must be clear and readable.
        raise HTTPException(422, str(exc)) from exc

    buffer = io.BytesIO()
    Image.fromarray(result.stego).save(buffer, format="PNG")
    buffer.seek(0)

    stem = Path(cover.filename or "cover").stem
    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="{stem}-carrier.png"',
            # Surfaced in the UI so the user sees what the embedding cost.
            "X-Sagex-Payload-Bytes": str(result.payload_bytes),
            "X-Sagex-Stored-Bytes": str(result.stored_bytes),
            "X-Sagex-Pixels-Changed": str(result.pixels_changed),
            "X-Sagex-Psnr": f"{result.psnr_db:.1f}",
            "Access-Control-Expose-Headers": "X-Sagex-Payload-Bytes, X-Sagex-Stored-Bytes, "
                                             "X-Sagex-Pixels-Changed, X-Sagex-Psnr",
        },
    )


@app.post("/api/reveal")
def api_reveal(
    stego: UploadFile = File(...),
    passphrase: str = Form(...),
    filename: str = Form("recovered.bin"),
):
    """Recover a hidden file from an image."""
    if not passphrase:
        raise HTTPException(400, "passphrase must not be empty")

    array = _decode_image(_read_upload(stego), stego.filename or "image")

    try:
        payload = reveal(array, passphrase)
    except FrameError as exc:
        raise HTTPException(404, str(exc)) from exc
    except DecryptionError as exc:
        raise HTTPException(403, str(exc)) from exc

    safe_name = Path(filename).name or "recovered.bin"
    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}"',
            "X-Sagex-Recovered-Bytes": str(len(payload)),
            "Access-Control-Expose-Headers": "X-Sagex-Recovered-Bytes",
        },
    )


@app.post("/api/rank")
def api_rank(
    images: list[UploadFile] = File(...),
    size: int = Form(...),
    rate: str = Form("balanced"),
):
    """Rank candidate covers by how well they would hide ``size`` bytes."""
    if size < 0:
        raise HTTPException(400, "size must not be negative")

    resolved = _resolve_rate(rate)
    candidates: dict[str, np.ndarray] = {}
    for upload in images:
        name = upload.filename or f"image-{len(candidates)}"
        candidates[name] = _decode_image(_read_upload(upload), name)

    ranked = capacity.rank_covers(candidates, size, resolved)
    ranked_names = {name for name, _ in ranked}
    rejected = [
        {"filename": name, "summary": capacity.analyse(array, resolved).summary()}
        for name, array in candidates.items()
        if name not in ranked_names
    ]
    return {
        "payload_bytes": size,
        "considered": len(candidates),
        "ranked": [{"filename": n, **_report_json(c)} for n, c in ranked],
        "rejected": rejected,
    }


@app.get("/api/health")
async def api_health():
    return {"status": "ok", "max_upload_bytes": MAX_UPLOAD_BYTES}


@app.get("/")
async def index():
    return FileResponse(_STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


def main() -> None:
    """Entry point for the ``sagex-web`` script."""
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
