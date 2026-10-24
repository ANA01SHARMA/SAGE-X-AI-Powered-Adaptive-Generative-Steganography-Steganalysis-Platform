"""EXP8: can a HiDDeN/TrustMark-style robust codec actually TRAIN on a 4GB 3050 Laptop?
Encoder + noise layer + decoder + discriminator, all resident, all backpropped together.
This is the measurement Mode B lives or dies on."""
import torch, torch.nn as nn, torch.nn.functional as F, gc, time, os
torch.set_num_threads(os.cpu_count())

def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class Encoder(nn.Module):
    """cover + replicated message -> residual. HiDDeN topology, TrustMark-ish width."""
    def __init__(s, msg=100, ch=64, blocks=4):
        super().__init__()
        s.msg = msg
        s.pre = cbr(3, ch)
        s.body = nn.Sequential(*[cbr(ch, ch) for _ in range(blocks)])
        s.fuse = cbr(ch + msg + 3, ch)
        s.out = nn.Conv2d(ch, 3, 1)
    def forward(s, x, m):
        h = s.body(s.pre(x))
        mm = m.view(m.size(0), s.msg, 1, 1).expand(-1, -1, x.size(2), x.size(3))
        return s.out(s.fuse(torch.cat([h, mm, x], 1)))

class Decoder(nn.Module):
    def __init__(s, msg=100, ch=64, blocks=6):
        super().__init__()
        s.body = nn.Sequential(cbr(3, ch), *[cbr(ch, ch) for _ in range(blocks)], cbr(ch, msg))
        s.head = nn.Linear(msg, msg)
    def forward(s, x):
        return s.head(F.adaptive_avg_pool2d(s.body(x), 1).flatten(1))

class Disc(nn.Module):
    def __init__(s, ch=32):
        super().__init__()
        s.b = nn.Sequential(cbr(3, ch), cbr(ch, ch, 2), cbr(ch, ch, 2), nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(), nn.Linear(ch, 1))
    def forward(s, x): return s.b(x)

def diff_jpeg(x, q=50):
    """differentiable JPEG surrogate: 8x8 DCT -> soft-quantise -> IDCT (the usual trick)."""
    B, C, H, W = x.shape
    # crude but representative cost-wise: blockwise DCT via unfold + matmul
    n = 8
    u = F.unfold(x.reshape(B * C, 1, H, W), n, stride=n)          # B*C, 64, L
    k = torch.arange(n, dtype=x.dtype, device=x.device)
    M = torch.cos((2 * k[None, :] + 1) * k[:, None] * torch.pi / (2 * n))
    M[0] /= (2 ** 0.5); M = M * (2.0 / n) ** 0.5
    blk = u.transpose(1, 2).reshape(-1, n, n)
    D = M @ blk @ M.t()
    scale = 50.0 / q
    Dq = D + (torch.round(D / scale) * scale - D).detach()         # STE
    r = M.t() @ Dq @ M
    r = r.reshape(B * C, -1, n * n).transpose(1, 2)
    return F.fold(r, (H, W), n, stride=n).reshape(B, C, H, W)

def noise_layer(x, mode):
    if mode == "jpeg":   return diff_jpeg(x, 50)
    if mode == "resize": return F.interpolate(F.interpolate(x, scale_factor=0.5, mode="bilinear",
                                 align_corners=False), size=x.shape[-2:], mode="bilinear", align_corners=False)
    if mode == "crop":
        B, C, H, W = x.shape; c = int(H * 0.8)
        o = (H - c) // 2
        return F.interpolate(x[:, :, o:o + c, o:o + c], size=(H, W), mode="bilinear", align_corners=False)
    if mode == "blur":
        k = torch.tensor([[1., 2, 1], [2, 4, 2], [1, 2, 1]], device=x.device) / 16
        return F.conv2d(x, k.expand(3, 1, 3, 3), padding=1, groups=3)
    return x + torch.randn_like(x) * 0.02

def measure(msg, ch, sz, bs, blocks=(4, 6), use_disc=True):
    enc, dec, dsc = Encoder(msg, ch, blocks[0]), Decoder(msg, ch, blocks[1]), Disc()
    acts = []
    mods = [m for M in (enc, dec, dsc) for m in M.modules() if not list(m.children())]
    hs = [m.register_forward_hook(lambda mo, i, o: acts.append(
            sum(t.numel() * t.element_size() for t in (o if isinstance(o, (tuple, list)) else [o])
                if torch.is_tensor(t)))) for m in mods]
    x = torch.rand(bs, 3, sz, sz); m = torch.randint(0, 2, (bs, msg)).float()
    t0 = time.time()
    res = enc(x, m); st = (x + res).clamp(0, 1)
    tot = 0
    for mode in ("jpeg", "resize", "crop", "blur", "noise"):      # all 5 in one step
        out = dec(noise_layer(st, mode))
        tot = tot + F.binary_cross_entropy_with_logits(out, m)
    if use_disc:
        tot = tot + 1e-3 * F.binary_cross_entropy_with_logits(dsc(st), torch.ones(bs, 1))
    tot = tot + 2.0 * F.mse_loss(st, x)
    tot.backward()
    dt = time.time() - t0
    for h in hs: h.remove()
    p = sum(q.numel() * q.element_size() for M in (enc, dec, dsc) for q in M.parameters()) / 2**20
    a = sum(acts) / 2**20 / 2          # AMP fp16
    est = p * 4 + a + 250
    del enc, dec, dsc, x, m; gc.collect()
    return p, a, est, dt

BUDGET_WIN, BUDGET_LIN = 3200, 3800
print("=" * 104)
print("EXP8  robust neural codec (HiDDeN/TrustMark topology) -- enc + 5 noise layers + dec + disc")
print("      train VRAM = params*4 (fp32 master+grad+2 Adam moments) + fp16 activations + 250MB workspace")
print("=" * 104)
print(f"{'config':<44}{'params':>9}{'acts':>9}{'VRAM':>9}{'CPU s/it':>10}{'3050 s/it':>11}  verdict")
print("-" * 104)
for msg, ch, sz, bs, blocks in [
        (100, 64, 128, 4,  (4, 6)),
        (100, 64, 128, 8,  (4, 6)),
        (100, 64, 256, 4,  (4, 6)),
        (100, 64, 256, 8,  (4, 6)),
        (100, 32, 256, 8,  (3, 4)),
        (100, 32, 256, 16, (3, 4)),
        (100, 64, 512, 2,  (4, 6)),
        (256, 64, 256, 4,  (4, 6))]:
    p, a, e, dt = measure(msg, ch, sz, bs, blocks)
    v = "FITS(win)" if e < BUDGET_WIN else ("Linux only" if e < BUDGET_LIN else "NO")
    g = dt / 20
    print(f"{f'msg={msg} ch={ch} {sz}px bs={bs} blk={blocks}':<44}"
          f"{p:>8.1f}M{a:>8.0f}M{e:>8.0f}M{dt:>10.2f}{g:>11.3f}  {v}")

print("\n-- convergence estimate (TrustMark/HiDDeN report ~300-500 epochs over ~100k crops) --")
for label, sz, bs, dt in [("ch=32 256px bs=16", 256, 16, None)]:
    pass
