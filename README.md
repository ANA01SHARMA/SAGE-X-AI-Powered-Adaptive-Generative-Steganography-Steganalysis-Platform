# SAGE-X

Hide an encrypted file inside an ordinary photograph. The photo looks unchanged, and
only someone with the passphrase can tell there is anything in it, or get it back.

What makes this different from every other steganography tool: **it works out whether
your data can actually be recovered before it hides anything, and refuses when it
can't.** Steghide, OpenStego and SilentEye all embed whatever you give them
and report success; you find out it failed only when extraction returns nothing.

## Start here

```bash
bash start.sh
```

Then open **http://127.0.0.1:3000** in a browser. Press Ctrl-C in the terminal to stop.

That is the whole thing. The first run builds a virtual environment and installs
dependencies, which takes a minute; after that it starts in a couple of seconds. It is
safe to run again any time.

If port 3000 is busy, either `bash stop.sh` to free it, or `bash start.sh 9000` to use
another port.

### Using the web interface

| tab | what it does |
|---|---|
| **Hide** | pick a photo and a file, get back a carrier image |
| **Reveal** | pick a carrier image, get your file back |
| **Analyse** | see how much a photo could hold, before committing to it |
| **Rank covers** | pick several photos, see which is the best hiding place |

The Hide tab tells you whether your file fits *as you choose it*, and will not let you
embed something that cannot be recovered.

## Command line

If you prefer the terminal, activate the environment first:

```bash
source .venv/bin/activate     # only needed once per terminal session

sagex capacity photo.png                          # what can this image hold?
sagex hide photo.png secret.pdf -o carrier.png    # hide a file
sagex reveal carrier.png -o recovered.pdf         # get it back
sagex rank ~/Pictures --size 50000                # which photo is the best cover?
```

Without activating, prefix each command with `.venv/bin/` instead.

Or from Python:

```python
from sagex import hide, reveal, imageio, capacity

cover = imageio.load("photo.png")
print(capacity.analyse(cover).summary())
# 5120x3200 x3 colour: holds up to 599.9 KB at the balanced rate.
# Image is moderately textured (texture score 0.35).

result = hide(cover, b"secret data", passphrase="correct horse battery staple")
imageio.save("carrier.png", result.stego)

recovered = reveal(imageio.load("carrier.png"), "correct horse battery staple")
```

## How it works

```
your file ──► compress ──► encrypt (AES-256-GCM) ──► frame ──► scatter into pixels ──► PNG
                                                                                        │
                                                                                    share it
                                                                                        │
your file ◄── decompress ◄── decrypt ◄── unframe ◄── read pixels ◄──────────────────────┘
```

Each bit is stored in the lowest bit of one colour value. If the bit already matches,
nothing changes; otherwise the value moves up or down by one. A value of 200 becomes
199 or 201, a difference no eye or screen can show.

**Which** pixels are used depends on your passphrase, so an attacker who does not know
it cannot even tell which pixels to examine.

### The bootstrapping problem

There is a chicken-and-egg problem worth understanding, because it shapes the whole
design. The pixel positions depend on the key. The key depends on a random salt. The
salt is itself hidden in the image. So something has to be readable *before* the key
exists.

The fix is two blocks:

| block | size | where | contents |
|---|---|---|---|
| bootstrap | 16 bytes | fixed positions, no key needed | the salt |
| payload | variable | key-derived positions | header + encrypted data |

The receiver reads the salt first, derives the key from it and the passphrase, and only
then knows where the real data lives.

The salt is public, not secret. It exists so the same passphrase produces
different pixel positions for every image. Without it, someone collecting several of
your images could line them up and spot that the same pixels keep changing.

## Capacity

Three limits stack up:

1. **Physical**: one bit per colour value. A 12-megapixel photo has ~36 million.
2. **Safety**: we use only a fraction of those. Using every pixel is trivially
   detectable. The rate is 0.05 (conservative), 0.1 (balanced) or 0.4 (aggressive).
3. **Overhead**: 54 bytes of salt, header, nonce and authentication tag, before any
   payload fits at all.

Small images fail at step 3. A 200×200 photo at the conservative rate has room for
fewer bytes than the overhead needs, so the honest answer is zero, and that is what
it says instead of producing a broken file.

| image | conservative | balanced | aggressive |
|---|---|---|---|
| 512×512 | 4.7 KB | 9.5 KB | 38 KB |
| 5120×3200 | 300 KB | 600 KB | 2.3 MB |

## Important limitation

**The image must arrive byte-for-byte identical.** WhatsApp, Instagram, Twitter and
iMessage all re-compress what you send, and that destroys every hidden bit. We
measured the recovery rate after JPEG compression at essentially zero, no better than
guessing.

Use email attachments, Google Drive, Dropbox, Telegram's "send as file", or Signal.
`sagex` refuses to save to a lossy format for this reason.

## Code layout

```
sagex/
├── capacity.py       what fits, and the refusal when nothing does
├── pipeline.py       hide() and reveal(), start reading here
├── imageio.py        loading and saving; rejects lossy formats
├── core/
│   ├── bitio.py      bytes <-> bits
│   ├── crypto.py     Argon2id key derivation, AES-256-GCM
│   └── frame.py      the byte layout, and why it has two blocks
└── embed/
    ├── costs.py      which pixels are safe to change (HILL)
    ├── lsbm.py       the actual bit-writing
    └── positions.py  which pixels carry which bits
```

`pipeline.py` is the best entry point. It is short and calls into everything else.

## Tests

```bash
pytest
```

Covers round trips for text, binary and empty payloads, colour and grayscale, wrong
passphrases, tampered images, capacity refusal, and the guarantee that no pixel ever
moves by more than one.

## Background

The design choices here follow from measurements in [`rnd/FINDINGS.md`](rnd/FINDINGS.md),
in particular why this tool does not claim to survive compression and why it does not
use generative AI. The scripts that produced those numbers are in `rnd/`.
