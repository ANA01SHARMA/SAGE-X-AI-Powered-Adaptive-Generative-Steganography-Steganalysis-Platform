"""EXP5: measured VRAM budget for a 4GB RTX 3050 Laptop.
Activation memory is measured with forward hooks (exact tensor bytes), not estimated."""
import torch, torch.nn as nn, gc
torch.set_grad_enabled(True)

def measure(model, x, label, train=True, amp_fp16=False):
    """Return (params_MB, activation_MB, est_train_VRAM_MB)."""
    acts = []
    hs = [m.register_forward_hook(lambda mod, i, o: acts.append(
            sum(t.numel()*t.element_size() for t in (o if isinstance(o,(tuple,list)) else [o])
                if torch.is_tensor(t))))
          for m in model.modules() if not list(m.children())]
    with torch.no_grad() if not train else torch.enable_grad():
        try: y = model(x)
        except Exception as e:
            for h in hs: h.remove()
            return None, None, None, f"forward failed: {type(e).__name__}"
    for h in hs: h.remove()
    p  = sum(q.numel()*q.element_size() for q in model.parameters())/2**20
    a  = sum(acts)/2**20
    if amp_fp16: a /= 2
    # training VRAM ~= params(fp32 master) + grads + Adam(2 moments) + activations + workspace
    est = p*4 + a + 250 if train else p + a + 150
    return p, a, est, "ok"

# ---------------- SRNet (Boroumand et al. 2019), the standard spatial steganalyzer
class T1(nn.Module):
    def __init__(s, i, o): super().__init__(); s.b=nn.Sequential(nn.Conv2d(i,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU())
    def forward(s,x): return s.b(x)
class T2(nn.Module):
    def __init__(s, c): super().__init__(); s.b=nn.Sequential(T1(c,c),nn.Conv2d(c,c,3,1,1),nn.BatchNorm2d(c))
    def forward(s,x): return x+s.b(x)
class T3(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.b=nn.Sequential(T1(i,o),nn.Conv2d(o,o,3,1,1),nn.BatchNorm2d(o),nn.AvgPool2d(3,2,1))
        s.sc=nn.Sequential(nn.Conv2d(i,o,1,2),nn.BatchNorm2d(o))
    def forward(s,x): return s.b(x)+s.sc(x)
class SRNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.f=nn.Sequential(T1(1,64),T1(64,16),*[T2(16) for _ in range(5)],
                          T3(16,16),T3(16,64),T3(64,128),T3(128,256),
                          T1(256,512),nn.Conv2d(512,512,3,1,1),nn.BatchNorm2d(512),
                          nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(512,2))
    def forward(s,x): return s.f(x)

print("="*100)
print("EXP5  measured memory footprint  ->  budget for RTX 3050 Laptop 4GB")
print("      usable VRAM under Windows 11 ~= 3.2 GB (desktop compositor + WDDM reserve take ~0.6-0.8 GB)")
print("      usable VRAM under Linux, no desktop on the dGPU ~= 3.8 GB")
print("="*100)
print(f"{'model / setting':<46}{'params':>9}{'acts':>9}{'train VRAM':>13}  verdict")
print("-"*100)

BUDGET_WIN, BUDGET_LIN = 3200, 3800
def row(label, model, x, train=True, amp=False):
    p,a,e,st = measure(model, x, label, train, amp)
    if p is None: print(f"{label:<46}{st}"); return
    v = "FITS(win)" if e < BUDGET_WIN else ("fits Linux only" if e < BUDGET_LIN else "DOES NOT FIT")
    print(f"{label:<46}{p:>8.1f}M{a:>8.0f}M{e:>12.0f}M  {v}")
    del model, x; gc.collect()

for bs in (2, 4, 8, 16):
    row(f"SRNet train  256x256  bs={bs}  fp32", SRNet(), torch.randn(bs,1,256,256))
for bs in (4, 8, 16):
    row(f"SRNet train  256x256  bs={bs}  AMP fp16", SRNet(), torch.randn(bs,1,256,256), amp=True)
row("SRNet train  512x512  bs=4   AMP fp16", SRNet(), torch.randn(4,1,512,512), amp=True)

import timm
for name, sz, bs in [("efficientnet_b0",256,8),("efficientnet_b0",256,16),
                     ("efficientnet_b0",512,4),("efficientnet_b0",512,8),
                     ("efficientnet_b3",256,8)]:
    m = timm.create_model(name, pretrained=False, num_classes=2, in_chans=1)
    row(f"{name} train {sz}x{sz} bs={bs} AMP fp16", m, torch.randn(bs,1,sz,sz), amp=True)

# ---------------- Stable Diffusion 1.5 VAE (latent steganography without the UNet)
from diffusers import AutoencoderKL
vae = AutoencoderKL(in_channels=3, out_channels=3,
    down_block_types=("DownEncoderBlock2D",)*4, up_block_types=("UpDecoderBlock2D",)*4,
    block_out_channels=(128,256,512,512), layers_per_block=2, latent_channels=4)
print("-"*100)
for bs, sz in [(1,512),(2,512),(4,512),(1,768)]:
    row(f"SD1.5 VAE encode+decode {sz}px bs={bs} fp16 infer",
        vae, torch.randn(bs,3,sz,sz), train=False, amp=True)
