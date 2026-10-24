# SAGE-X — High-Level Design

Scope per `SCOPE.md`; every constraint traced to a measurement in `FINDINGS.md`.

---

## 1. Architectural drivers

The R&D dictates the architecture. These are not preferences.

| # | driver | source | consequence |
|---|---|---|---|
| **D1** | The demo must never depend on the 3050 being alive | viva risk | **CPU-first.** GPU is an accelerator, never a dependency. |
| **D2** | 8 GB system RAM, ~3–4 GB usable | F7 | **One** ML worker process, models loaded once. Not N workers × PyTorch. |
| **D3** | Robust and covert modes are mutually exclusive | F1, F4, F6 | Two parallel pipelines, shared crypto/ECC. Not one "adaptive" engine. |
| **D4** | Capacity depends on image **and** channel | F9 | The workbench is an **online service**, not an offline test harness. |
| **D5** | Jobs run seconds→minutes (adaptive loop, robustness suite) | F3, 2.1 | Async job API + progress streaming. No synchronous embed endpoint. |
| **D6** | 512² @ q50 cannot hold even the GCM header | F9 | **Capacity check before embed**, fail fast with the number. |

---

## 2. Layered view

```
┌──────────────────────────────────────────────────────────────────┐
│ PRESENTATION   React 18 + TS · Vite · TanStack Query · Recharts  │
├──────────────────────────────────────────────────────────────────┤
│ API            FastAPI · REST + WebSocket job stream · JWT       │
├──────────────────────────────────────────────────────────────────┤
│ ORCHESTRATION  Celery · queue:cpu (n=2) · queue:ml (n=1) · Redis │
├──────────────────────────────────────────────────────────────────┤
│ DOMAIN                                                           │
│   ┌────────────┬────────────┬───────────────┬────────────────┐   │
│   │ Mode A     │ Mode B     │ Steganalysis  │ Adaptive loop  │   │
│   │ covert     │ robust     │ detectors     │ ADV-EMB        │   │
│   │ HILL + STC │ codec/QIM  │ EffNet + SPAM │ (Mode A only)  │   │
│   └─────┬──────┴─────┬──────┴───────┬───────┴────────┬───────┘   │
│   ┌─────┴────────────┴──────────────┴────────────────┴───────┐   │
│   │ SHARED   crypto · ECC · payload framing · workbench      │   │
│   │          attacks · metrics · frontier                    │   │
│   └──────────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────┤
│ PERSISTENCE    PostgreSQL (metadata)  ·  MinIO / FS (blobs)      │
└──────────────────────────────────────────────────────────────────┘
```

**Why the shared layer sits below both modes:** crypto, ECC, framing, the attack suite and
the metrics are mode-agnostic. Only the *embedding* differs. This is what makes the frontier
comparison meaningful — both modes are measured by identical instrumentation.

---

## 3. Module decomposition

```
sagex/
├── core/
│   ├── crypto.py        AES-256-GCM · Argon2id KDF · nonce from cover hash
│   ├── ecc.py           BCH / repetition · rate selection from target BER
│   └── payload.py       header framing · CRC · version · length
├── modes/
│   ├── covert/          Mode A
│   │   ├── costs.py     HILL · S-UNIWARD · MiPOD cost maps
│   │   ├── stc.py       syndrome-trellis coding (fallback: Hamming matrix embedding)
│   │   └── simulator.py optimal-embedding simulator (for capacity bounds)
│   └── robust/          Mode B
│       ├── qim.py       block-DCT QIM  ← guaranteed fallback (F3)
│       └── codec/       encoder · noise layers · decoder · discriminator
├── analysis/
│   ├── detectors/
│   │   ├── effnet.py    EfficientNet-B0, GPU or CPU
│   │   └── spam.py      SPAM-686 + FLD ensemble ← fast CPU second opinion
│   ├── calibration.py   score → calibrated P(stego)
│   └── adaptive.py      ADV-EMB closed loop
├── bench/
│   ├── attacks.py       12-attack suite (JPEG·WebP·resize·crop·blur·noise)
│   ├── metrics.py       BER · PSNR · SSIM · P_E
│   └── frontier.py      δ sweep · capacity curve · risk score
├── api/routers/         assets · capacity · embed · extract · analyze · robustness · jobs
└── workers/tasks.py     Celery task definitions
```

`bench/` is shared by the research CLI and the web API — **one implementation**, so the
dashboard numbers and the report numbers can never diverge.

---

## 4. Key sequences

### S1 — Embed (Mode A, with optional adaptive loop)

```
Client          API           queue:cpu        queue:ml        Store
  │              │                │               │              │
  ├─ POST /embed ►                │               │              │
  │              ├─ validate      │               │              │
  │              ├─ store cover ──────────────────────────────── ►
  │              ├─ Job(queued) ─ ►               │              │
  ◄─ 202 job_id ─┤                │               │              │
  │              │                ├ derive key (Argon2id)        │
  │              │                ├ AES-256-GCM encrypt          │
  │              │                ├ capacity check ◄── D6        │
  │              │                │   payload > capacity?        │
  │              │                │   └─► FAIL with max_bytes    │
  │              │                ├ ECC encode                   │
  │              │                ├ HILL cost map                │
  │              │                ├ STC embed                    │
  │              │                ├ PSNR/SSIM + SPAM risk        │
  │              │                │               │              │
  │           if adaptive:        ├── enqueue ──► ┤              │
  ├─ WS /jobs/{id}/stream ────────────────────────┤              │
  ◄─ iter 1..8 progress ──────────────────────────┤ ADV-EMB loop │
  │              │                │               ├ store stego ►│
  ◄─ done + metrics ──────────────┴───────────────┘              │
```

### S2 — Steganalysis

```
upload ─► Job ─► queue:ml ─┬─ EfficientNet-B0 → score
                           ├─ SPAM+FLD (CPU)  → second opinion
                           ├─ calibration     → P(stego)
                           └─ report: verdict · confidence
                                    · nearest-matching algorithm
                                    · MISMATCH CAVEAT (always shown)
```

The mismatch caveat is a **product requirement**, not a footnote: the detector scores
88–93% on algorithms it was trained on and 60–70% on ones it wasn't. A risk report that
hides that is dishonest, and saying so out loud is a viva asset.

### S3 — Robustness test / frontier

```
embedding_id ─► queue:cpu ─► for each of 12 attacks:
                                 attacked = attack(stego)
                                 bits     = extract(attacked)
                                 BER, recovered, PSNR
                             ─► δ sweep → frontier curve
                             ─► cache by (sha256, channel)
                             ─► table + chart payload
```

---

## 5. Data model

Blobs never go in Postgres — metadata only, paths into object storage.

| table | key columns |
|---|---|
| `users` | id · email · pw_hash · created_at |
| `assets` | id · user_id · kind{cover,stego,attacked} · sha256 · path · w · h · bytes |
| `jobs` | id · user_id · type · status · params `jsonb` · result `jsonb` · error · timestamps |
| `embeddings` | id · job_id · cover_id · stego_id · mode{A,B} · algo · bpp · payload_bytes · psnr · ssim · pe_estimate · adaptive_iters |
| `attack_results` | id · embedding_id · attack · param · ber · recovered · psnr |
| `detections` | id · asset_id · detector · raw_score · calibrated_p · verdict |
| `frontier_cache` | sha256 · channel · curve `jsonb` — the δ sweep is expensive, cache it |

---

## 6. API surface

```
POST  /api/v1/auth/register | /login
POST  /api/v1/assets                    multipart → asset_id
GET   /api/v1/assets/{id}[/download]

POST  /api/v1/capacity                  {asset_id, channel, mode}
                                     →  {max_bpp, max_bytes, pe_curve[]}      ← D6
POST  /api/v1/embed                     {asset_id, mode, payload, passphrase,
                                         channel, adaptive:bool}          → 202 job
POST  /api/v1/extract                   {asset_id, mode, passphrase}      → 202 job
POST  /api/v1/analyze                   {asset_id}                        → 202 job
POST  /api/v1/robustness                {embedding_id, attacks[]}         → 202 job

GET   /api/v1/jobs/{id}                 status + result
WS    /api/v1/jobs/{id}/stream          progress events (adaptive iterations)  ← D5

GET   /api/v1/frontier?asset_id=&channel=    curve for the dashboard
GET   /api/v1/experiments                    research aggregates
```

`/capacity` being a **separate call before** `/embed` is the architectural expression of F9.
The UI calls it on channel-select and drives the payload-size slider from the response, so
the user is told what fits before they type a secret that cannot.

---

## 7. Deployment view

```
docker-compose
├── postgres      ~200 MB
├── redis         ~50 MB
├── minio         ~150 MB
├── api           ~200 MB    uvicorn, no models
├── worker-cpu    ~250 MB ×2  Mode A · attacks · metrics   (pure numpy)
├── worker-ml     ~600 MB ×1  detectors · adaptive · Mode B   ← D2, concurrency=1
└── web           static
                  ─────────
                  ~1.7 GB total → fits 8 GB with headroom
```

`worker-ml` probes for CUDA at boot and falls back to CPU on the same code path — so the
identical compose file runs on the 3050 laptop, on the i7 dev box, and on a marker's machine.

---

## 8. Non-functional requirements

| NFR | target | basis |
|---|---|---|
| Mode A embed, 512² | < 500 ms | 200 ms measured, CPU |
| Steganalysis inference | < 300 ms CPU · < 20 ms GPU | F8 |
| Robustness suite, 12 attacks | < 20 s | 18 s measured (EXP1) |
| Adaptive loop, 8 iterations | < 30 s | ~1 s/iter + overhead |
| Concurrent demo users | 5–10 | single box |
| Max upload | 12 MB / 4096² | RAM bound, D2 |
| Peak RSS, all services | < 3 GB | leaves headroom on 8 GB |

---

## 9. Design decisions (ADR summary)

| # | decision | rationale |
|---|---|---|
| ADR-1 | CPU-first; GPU optional | D1 — demo survives a dead GPU |
| ADR-2 | Two modes, not one unified engine | F4 — the frontier is a step function |
| ADR-3 | Capacity check precedes embed | F9 — 512²@q50 can't hold the header |
| ADR-4 | Workbench is an online service | D4 — it drives the risk slider |
| ADR-5 | Blobs in object storage, not Postgres | image sizes |
| ADR-6 | Single ML worker, concurrency 1 | D2 — 8 GB RAM |
| ADR-7 | Mode B behind a feature flag, QIM fallback | 2.2 is the only high-risk item |
| ADR-8 | `bench/` shared by CLI and API | dashboard and report can't diverge |

---

## 10. Threat model

**Mode A — covert channel.** Adversary is a passive warden running a steganalyzer.
Channel is assumed lossless (PNG). Security claim: P_E vs a *matched* CNN at the configured
bpp, reported per-image. No claim against an adaptive adversary who retrains on our output.

**Mode B — robust watermark.** Adversary is **JPEG and a resize**, not a censor. Detection
is expected and acceptable. Security claim: bit accuracy under the declared channel set.
No covert-channel claim whatsoever.

Confusing these two threat models is the single most common error in this literature, and
keeping them architecturally separate is the point of ADR-2.
