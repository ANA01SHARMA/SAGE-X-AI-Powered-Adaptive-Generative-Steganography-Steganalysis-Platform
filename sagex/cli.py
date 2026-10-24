"""Command-line interface.

    sagex capacity photo.png
    sagex hide photo.png secret.pdf -o carrier.png
    sagex reveal carrier.png -o recovered.pdf
    sagex rank ~/Pictures --size 50000

The passphrase is prompted for rather than passed as an argument, because anything on
the command line ends up in shell history and in the process list.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from . import capacity, imageio
from .core.crypto import DecryptionError
from .core.frame import FrameError
from .pipeline import hide, reveal

_RATES = {
    "conservative": capacity.RATE_CONSERVATIVE,
    "balanced": capacity.RATE_BALANCED,
    "aggressive": capacity.RATE_AGGRESSIVE,
}

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _ask_passphrase(confirm: bool = False) -> str:
    passphrase = getpass.getpass("Passphrase: ")
    if not passphrase:
        raise SystemExit("error: passphrase must not be empty")
    if confirm and getpass.getpass("Confirm passphrase: ") != passphrase:
        raise SystemExit("error: passphrases do not match")
    return passphrase


def _cmd_capacity(args: argparse.Namespace) -> int:
    image = imageio.load(args.image)
    for name, rate in _RATES.items():
        report = capacity.analyse(image, rate)
        marker = " <- default" if rate == capacity.DEFAULT_RATE else ""
        print(f"{name:>13}: {report.summary()}{marker}")
    return 0


def _cmd_hide(args: argparse.Namespace) -> int:
    cover = imageio.load(args.cover)
    payload = Path(args.payload).read_bytes()

    try:
        result = hide(cover, payload, _ask_passphrase(confirm=True), rate=_RATES[args.rate])
    except capacity.InsufficientCapacity as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    imageio.save(args.output, result.stego)
    print(result.summary())
    print(f"Written to {args.output}")
    print("Send this file as-is. Re-saving it as JPEG will destroy the payload.")
    return 0


def _cmd_reveal(args: argparse.Namespace) -> int:
    stego = imageio.load(args.stego)
    try:
        payload = reveal(stego, _ask_passphrase())
    except FrameError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except DecryptionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    Path(args.output).write_bytes(payload)
    print(f"Recovered {len(payload):,} bytes to {args.output}")
    return 0


def _cmd_rank(args: argparse.Namespace) -> int:
    folder = Path(args.folder)
    candidates = {
        path.name: imageio.load(path)
        for path in sorted(folder.iterdir())
        if path.suffix.lower() in _IMAGE_SUFFIXES
    }
    if not candidates:
        print(f"error: no images found in {folder}", file=sys.stderr)
        return 1

    ranked = capacity.rank_covers(candidates, args.size, _RATES[args.rate])
    print(f"{len(ranked)} of {len(candidates)} images can hold {args.size:,} bytes "
          f"at the {args.rate} rate.\n")
    for position, (name, report) in enumerate(ranked[: args.top], start=1):
        print(f"{position:>2}. {name}")
        print(f"    {report.summary()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sagex", description="Hide encrypted data inside images."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    rate_kwargs = {"choices": list(_RATES), "default": "balanced",
                   "help": "embedding rate; lower is harder to detect (default: balanced)"}

    p = sub.add_parser("capacity", help="report what an image can hold")
    p.add_argument("image")
    p.set_defaults(func=_cmd_capacity)

    p = sub.add_parser("hide", help="hide a file inside an image")
    p.add_argument("cover")
    p.add_argument("payload", help="file to hide")
    p.add_argument("-o", "--output", required=True, help="stego image to write (.png)")
    p.add_argument("--rate", **rate_kwargs)
    p.set_defaults(func=_cmd_hide)

    p = sub.add_parser("reveal", help="recover a file from an image")
    p.add_argument("stego")
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=_cmd_reveal)

    p = sub.add_parser("rank", help="rank candidate covers for a payload size")
    p.add_argument("folder")
    p.add_argument("--size", type=int, required=True, help="payload size in bytes")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--rate", **rate_kwargs)
    p.set_defaults(func=_cmd_rank)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
