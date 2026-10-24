# SAGE-X — R&D feasibility findings

All numbers below are **measured on this machine**, not quoted from papers.
Repro: `cd rnd && ../.venv/bin/python exp1_robustness.py` (etc.)
Corpus: 24×512² + 500×256² grayscale tiles cropped from real photographs.

## Hardware reality

| | dev box (this one) | target box |
|---|---|---|
| CPU | i7-1355U, 12 threads | i5-12xxxH |
| GPU | Intel UHD (**no CUDA**) | RTX 3050 **Laptop** (GA107), 4 GB |
| RAM | 15 GB | 8 GB |

Usable VRAM on the 3050: **~3.2 GB under Windows 11** (WDDM reserve + compositor take
0.6–0.8 GB), **~3.8 GB under Linux** with the desktop on the iGPU. 8 GB system RAM under
Win11 leaves ~3–4 GB — that is the real bottleneck for dataloading, not VRAM.

---

## F1 — Spatial LSB dies on every channel (EXP1)

LSB-matching, 0.4 bpp, PSNR 55.1 dB:

| attack | BER |
|---|---|
| clean | 0.0000 |
| JPEG q95 | 0.4924 |
| WebP q90 | 0.4986 |
| resize .75 | 0.4733 |
| blur σ=0.5 | 0.5005 |

BER 0.49 = coin flip. **Zero** information survives, even at q95, even with a 256-bit
payload. Any design that hides bits in the spatial domain and then promises JPEG
robustness is dead on arrival.

## F2 — JPEG-DCT embedding survives only its own quality factor (EXP1)

F5-style LSB of quantised AC coefficients, QF90-locked, 4096 bits:
clean 0.0018 · q95 0.0146 · **q90 0.0017** · q75 0.7128 · q50 0.4587 · WebP q90 0.2694.

Works perfectly at the QF it was built for, collapses one step away. Brittle.

## F3 — Robust block-DCT QIM works, at ~64 bits (EXP1)

B=32, δ=18, rep=4 → 64 payload bits per 512² image, PSNR 56.6 dB:
clean/q95/q90/q75 = 0.000 · q50 0.027 · WebP q80 0.037 · resize .50 0.020 · noise 0.002.
**crop 90% = 0.463 (destroyed).**

## F4 — The security/robustness curve is a step function, not a trade-off (EXP3)

Sweeping δ, 64 payload bits. P_E vs SPAM-686 + FLD ensemble (P_E 0.50 = undetectable):

| δ | PSNR | P_E | JPEG q75 BER | JPEG q50 BER |
|---|---|---|---|---|
| 6 | 99.0 | 0.479 | 0.523 | 0.479 |
| 9 | 72.2 | 0.422 | 0.469 | 0.484 |
| 12 | 61.8 | 0.411 | 0.150 | 0.275 |
| **18** | 56.6 | **0.380** | **0.000** | **0.025** |
| 26 | 54.1 | 0.339 | 0.000 | 0.000 |
| 40 | 50.4 | 0.333 | 0.000 | 0.000 |

δ≤9 is not an operating point — PSNR 99 means the modification vanished under uint8
rounding, so the payload never existed. The usable window is δ 12→18, and it is **one
step wide**: the first δ that decodes reliably is already the first δ that a *weak,
1970s-era* detector catches. SRNet/EfficientNet would be far better than SPAM.

There is no δ that is both robust and undetectable. The "adaptive feedback loop" in the
proposal has essentially no search space to move in — **in the robust regime**.

## F5 — Crop is unrecoverable by search (EXP4)

16-bit sync preamble, brute-force scale×offset resync (15 scales × 64 offsets):

| | naive BER | after resync | preamble |
|---|---|---|---|
| no crop | 0.000 | 0.000 | 16.0/16 |
| crop 95% | 0.505 | 0.544 | 13.2/16 |
| crop 90% | 0.521 | 0.479 | 13.5/16 |

**Null test: a clean cover with nothing embedded scores 13.25/16 on the same search.**
The crop "detections" were search false positives. Resync found nothing. Bicubic
resampling destroys the QIM lattice even at perfect alignment.

→ Crop robustness is the one place a neural codec genuinely earns its keep (a decoder
trained with crop in the augmentation loop, HiDDeN/TrustMark style), or a Fourier–Mellin
geometric template. It is not reachable with hand-built DCT methods.

## F6 — Adaptive embedding IS undetectable, at high capacity, with zero robustness (EXP2)

300 cover/stego pairs @256², SPAM-686 + FLD ensemble:

| method | payload | P_E |
|---|---|---|
| LSB-matching 0.40 bpp | 26214 b | 0.246 broken |
| LSB-matching 0.10 bpp | 6553 b | 0.338 weak |
| **HILL adaptive 0.40 bpp** | **26214 b** | **0.483 secure** |
| HILL adaptive 0.10 bpp | 6553 b | 0.496 secure |
| HILL adaptive 0.01 bpp | 655 b | 0.500 secure |
| robust QIM δ=18 | 64 b | 0.375 weak |
| robust QIM δ=40 | 16 b | 0.412 marginal |

**Two orders of magnitude apart**: 26 214 undetectable-but-fragile bits vs 64
robust-but-detectable bits. These are two different products.

## F7 — Measured VRAM budget (EXP5/EXP6)

Activations measured with forward hooks (exact tensor bytes), not estimated.
Train VRAM = params×4 (fp32 master + grad + 2 Adam moments) + activations + 250 MB workspace.

| config | params | acts | train VRAM | 3.2 GB budget |
|---|---|---|---|---|
| SRNet 256² bs=8 AMP | 18.2 M | 903 M | 1226 M | fits |
| SRNet 256² bs=16 AMP | 18.2 M | 1806 M | 2129 M | fits |
| SRNet 512² bs=4 AMP | 18.2 M | 1806 M | 2129 M | fits |
| EfficientNet-B0 256² bs=16 AMP | 15.3 M | 907 M | 1218 M | fits |
| EfficientNet-B0 512² bs=8 AMP | 15.3 M | 1812 M | 2124 M | fits |
| EfficientNet-B3 256² bs=8 AMP | 40.8 M | 891 M | 1304 M | fits |
| **SD1.5 VAE, backprop, 512² bs=1** | 319 M | **4950 M** | **5419 M** | **does not fit** |

SD1.5 VAE **inference** at 512², fp16, measured peak ≈ 0.5–1.0 GB → fits fine.
(Caveat: measured via `ru_maxrss` high-water mark, so the tiled figure is unreliable;
the non-tiled 512² number is the trustworthy one.)

**Conclusion: the entire steganalysis half fits comfortably. Fine-tuning any latent
diffusion component does not — the VAE alone needs 5.4 GB just for retained activations.
Use the VAE frozen, at inference only, or not at all.**

## F8 — Throughput (EXP6b)

Measured on this CPU, scaled ×20 to a 3050 Laptop with AMP:

| config | CPU s/it | 3050 s/it | img/s | 1 epoch over 100k images |
|---|---|---|---|---|
| SRNet 256² bs=8 | 3.26 | 0.163 | 49 | 0.57 h |
| EffNet-B0 256² bs=16 | 4.32 | 0.216 | 74 | 0.37 h |
| EffNet-B0 512² bs=8 | 3.69 | 0.184 | 43 | 0.64 h |

SRNet from scratch needs ~400 k iterations to converge → **~18 h/epoch-equivalent, days
of wall-clock**. ImageNet-pretrained EfficientNet-B0 converges in ~30 epochs → **~11 h**.
Use the pretrained model. Do not train SRNet from scratch on this hardware.

## F9 — End-to-end payload budget

C = 1 − H(p); short-blocklength codes realistically reach ~55% of Shannon.

| channel | BER | C | 512² | 1024² | 2048² |
|---|---|---|---|---|---|
| JPEG q90 | 0.0034 | 0.967 | 136 b | 545 b | 2179 b |
| JPEG q75 | 0.0347 | 0.783 | 110 b | 441 b | 1764 b |
| JPEG q50 | 0.1177 | 0.477 | 67 b | 269 b | 1075 b |

Minus unavoidable crypto overhead (GCM nonce 96 + tag 128 + header 16 + CRC 32 = 272 b):

* 512² @ q50 → **negative**. Cannot carry even the header.
* 1024² @ q50 → ~0 bytes.
* 2048² @ q50 → **~100 bytes**.

Trim to a 64-bit tag, nonce derived from image hash, no CRC (GCM covers integrity) → 80 b
overhead → 1024² delivers ~23 bytes, 2048² ~124 bytes.

**The robust channel is a ~100-byte identifier channel, not a file-hiding channel.**
That is precisely what TrustMark and SynthID are (100–136 bits) — and they are the
state of the art. Anyone promising robust *and* high-capacity *and* undetectable is
promising three things that the measurements above show cannot coexist.
