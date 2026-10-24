# SAGE-X — realistic achievable scope

Derived from the measurements in `FINDINGS.md`. Every target below is either measured on
this hardware or taken from published work on comparable hardware. Nothing here is aspirational.

---

## The reframe

The proposal claims robust **and** high-capacity **and** undetectable. F1–F6 show those
three cannot coexist. Rather than quietly under-delivering on one, **ship two explicitly
separate modes and make the frontier between them the contribution.**

Old claim: *"AI-powered adaptive steganography that resists detection and compression."*
Defensible claim: *"A platform that measures, for a given image and a given channel, exactly
which payload sizes are achievable at what detection risk — and demonstrates empirically that
the robustness/undetectability frontier is a step function, not a trade-off curve."*

Nobody ships the second one. It is honest, it is novel, and every experiment backing it
already runs on your hardware.

---

## Tier 1 — will work, plan on delivering all of it

### 1.1 Robustness & detectability workbench *(the spine)*
Already ~60% built in `rnd/`. Harness that takes any embed/extract pair, runs the 12-attack
suite, runs the steganalyzer, emits the δ-sweep frontier curve.
**Effort:** 2–3 wks · **Hardware:** CPU only · **Risk:** none

### 1.2 Mode A — covert channel (high capacity, fragile)
HILL / S-UNIWARD cost map + STC coding. If STC proves fiddly, fall back to matrix embedding
(Hamming) — costs ~15% efficiency, saves a week.

| metric | target | basis |
|---|---|---|
| capacity @ 512² gray, 0.4 bpp | 13.1 KB | measured |
| capacity @ 512² RGB | ~39 KB | ×3 channels |
| P_E vs SPAM-686 @ 0.4 bpp | **0.483** | measured (F6) |
| P_E vs trained CNN @ 0.4 bpp | **~0.10–0.15** | literature (SRNet/BOSSBase) |
| P_E vs trained CNN @ 0.05–0.1 bpp | ~0.35–0.42 | literature |
| embed time, 512² | ~200 ms | measured, CPU |
| channel tolerance | **PNG only, zero JPEG** | measured (F1) |

**Set the product default at 0.05–0.1 bpp (1.6–3.3 KB), not 0.4.** The SPAM result is a
floor, not a guarantee — a CNN eats 0.4 bpp. Expose bpp as a user-facing risk slider driven
by 1.1's curve.

**Effort:** 3–4 wks · **Hardware:** CPU only · **Risk:** low

### 1.3 Steganalysis module
EfficientNet-B0, ImageNet-pretrained, 256², bs=16, AMP.

| metric | target | basis |
|---|---|---|
| train VRAM | **1218 MB** | measured (F7) |
| convergence | ~30 epochs ≈ 11 h | measured throughput (F8) |
| accuracy, matched algo+payload+QF | 88–93% | literature + ALASKA2 |
| accuracy, **unseen algorithm** | **60–70%** | literature |
| accuracy, cover-source mismatch | further −10–15 pts | literature |
| inference, 256² | ~15 ms GPU / ~200 ms CPU | measured |

**Report the mismatched number prominently.** "Our detector gets 91%" is a weak claim;
"our detector gets 91% matched and 64% on an algorithm it has never seen, and here is why"
is a real finding and a much better viva answer.

**Do not train SRNet from scratch** — ~400 k iterations, days of wall-clock (F8).

**Effort:** 3–4 wks · **Hardware:** Kaggle T4, or 3050 overnight · **Risk:** low

### 1.4 Crypto layer
AES-256-GCM, Argon2id KDF, nonce derived from cover hash. Keep the tag at 128 bits for
Mode A (capacity is plentiful); drop to 64 bits for Mode B where every bit costs.
**Effort:** 3 days · **Risk:** none

### 1.5 Full-stack platform
React + FastAPI + Celery/Redis + Postgres. **Everything except Mode B training runs on CPU**,
so the demo does not depend on the 3050 being alive.
**Effort:** 5–6 wks, parallelisable with the ML work · **Risk:** low

---

## Tier 2 — the research contribution, attempt with a fallback

### 2.1 Adaptive feedback loop — Mode A only
ADV-EMB (Tang et al., TIFS 2019): backprop through your own steganalyzer, scale the HILL
cost map by the gradient sign, re-embed, re-test, iterate.

| metric | target | basis |
|---|---|---|
| P_E vs the **targeted** detector | 0.48 → **0.49+** | published ADV-EMB results |
| P_E vs a **different** detector | **little or no gain** | known transferability limit |
| iterations to converge | 3–8 | published |
| cost per iteration | ~1 s | fwd+bwd on one image |

This is the one place the feedback loop in the proposal genuinely works — because Mode A has
a continuous, high-dimensional cost map to move in. **Report the transferability failure.**
It is the honest and more interesting half of the result.

**Effort:** 3–4 wks · **Risk:** medium (the gain is real but modest)

### 2.2 Mode B — robust neural codec
HiDDeN/TrustMark topology: encoder + differentiable noise layer + decoder + discriminator.
**The only route to crop robustness** (F5 proved search-based resync is impossible).

Config that fits 3.2 GB — **estimated, not measured; EXP8 was blocked before it ran:**

| knob | value | why |
|---|---|---|
| width | ch=32 | ch=64 @256² ≈ 2.9 GB, too close to the ceiling |
| resolution | 128–256 px | activations scale with area |
| batch | 8 | with grad accumulation to an effective 32 |
| noise layers | **1 sampled per step, not all 5** | 5-per-step multiplies decoder activations ×5 |
| est. train VRAM | ~1.8 GB @ ch=32/256²/bs=8 | analytic, same method that matched F7 |

**Verify this empirically before committing** — it is the one unmeasured number in this plan.

| metric | target | basis |
|---|---|---|
| payload | 100 bits raw | TrustMark/SynthID parity |
| bit acc @ JPEG q75 | ≥95% | published |
| bit acc @ JPEG q50 | ≥90% | published |
| bit acc @ crop 80% | ≥85% | published |
| **usable payload after ECC @ 512²** | **~6–8 bytes** | F9 |
| **usable payload after ECC @ 1024²** | **~16–23 bytes** | F9 |
| detectability | **will be detected — by design** | F4 |
| training | 2–4 wks of Kaggle iterations | 30 GPU-h/wk |

**Reframe Mode B's threat model.** It is a provenance watermark, not a covert channel. The
adversary is JPEG and a resize, not a censor running SRNet. Judged as a watermark it is
competitive with SOTA; judged as steganography it fails, and so does everyone else's.

**Fallback if it does not converge:** ship the hand-built robust QIM from F3 (64 bits,
survives everything but crop). Mode A is the guaranteed deliverable either way.

**Effort:** 4–5 wks · **Risk:** high — the only genuinely risky item

---

## Tier 3 — cut, with reasons

| item | why it dies |
|---|---|
| Training / fine-tuning any latent diffusion component | SD1.5 VAE needs **5419 MB** at bs=1 (F7). Not close. |
| Stable Diffusion cover generation in-app | Inference fits (~2.5–3.5 GB) but runs ~30–60 s/image on a mobile 3050. If you want generated covers, **batch-generate a fixed pool offline on Kaggle** and ship it. |
| SRNet trained from scratch | Days of wall-clock (F8). Use pretrained EfficientNet-B0. |
| "Semantic segmentation to find hiding regions" | HILL cost maps already do this better, for free, in ~50 ms of CPU. A segmentation model adds a GPU dependency and measurably *worse* security. **Reframe: the cost map IS the semantic map.** |
| Robust + high-capacity + undetectable simultaneously | F1, F4, F9. Physically unavailable. |
| Crop robustness via geometric search | F5 — null test proved the "detections" were false positives. Needs a learned decoder. |

---

## Environment decisions that buy more than any code optimisation

1. **Dual-boot Linux on the 3050 box.** Reclaims ~0.6 GB of VRAM from the WDDM reserve
   (3.2 → 3.8 GB) and several hundred MB of the 8 GB system RAM. Biggest single win available.
2. **8 GB system RAM is the real bottleneck, not VRAM.** Win11 leaves 3–4 GB for dataloading.
   Pre-decode the dataset to a memory-mapped uint8 array; use `num_workers=2`, never the default.
3. **Kaggle for all training runs** — 30 GPU-h/week on a T4 16 GB, free. Keep the 3050 for
   inference, debugging and demos.
4. **This dev box (i7-1355U, 15 GB, no CUDA) is the better machine for Tier 1.** More RAM than
   the 3050 laptop, and Mode A + the workbench are CPU-only anyway.

---

## Timeline — 24 weeks

| wk | track A (ML) | track B (platform) |
|---|---|---|
| 1–2 | env, Linux dual-boot, BOSSBase 1.01 (~1.5 GB), Kaggle setup | repo, CI, schema |
| 3–6 | **1.1 workbench** + **1.2 Mode A** | FastAPI skeleton, Celery, auth |
| 7–10 | **1.3 steganalysis** (Kaggle) | React shell, upload/embed/extract flows |
| 11–14 | **2.1 adaptive loop** ← research contribution | steganalysis + risk report UI |
| 15–19 | **2.2 Mode B** (high risk, fallback ready) | robustness-test UI, frontier dashboard |
| 20–22 | consolidation, ablations, mismatched-detector study | polish, deploy |
| 23–24 | paper / report, demo, viva prep | — |

**Critical path:** 1.1 → 1.3 → 2.1. Mode B (2.2) is deliberately off the critical path
so a failure there costs a feature, not the project.

---

## What to claim at the viva

- ✅ Two working modes with measured, honest operating envelopes
- ✅ A steganalyzer with matched **and** mismatched accuracy reported
- ✅ A working closed-loop adaptive embedder, with its transferability limit quantified
- ✅ **Empirical proof that the security/robustness frontier is a step function** — the
  result that most of this literature's marketing quietly assumes away
- ❌ Not: "undetectable", "unbreakable", "robust high-capacity steganography"

The negative results are the strongest part of this project. Lead with them.
