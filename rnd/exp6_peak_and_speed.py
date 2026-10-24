"""EXP6: (a) real inference peak for the SD VAE (tiled vs not) - measured as RSS delta.
        (b) training throughput, measured on this CPU then scaled to an RTX 3050 Laptop."""
import os, gc, time, torch, resource, numpy as np
torch.set_num_threads(os.cpu_count())

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024  # MB

from diffusers import AutoencoderKL
vae = AutoencoderKL(in_channels=3,out_channels=3,
    down_block_types=("DownEncoderBlock2D",)*4, up_block_types=("UpDecoderBlock2D",)*4,
    block_out_channels=(128,256,512,512), layers_per_block=2, latent_channels=4).eval()

print("="*92)
print("EXP6a  SD1.5 VAE actual INFERENCE peak (no_grad frees activations as it goes)")
print("="*92)
for tiling in (False, True):
    if tiling: vae.enable_tiling(); vae.enable_slicing()
    else:
        try: vae.disable_tiling(); vae.disable_slicing()
        except Exception: pass
    for sz in (512, 768):
        gc.collect(); base = rss()
        t0=time.time()
        with torch.no_grad():
            lat = vae.encode(torch.randn(1,3,sz,sz)).latent_dist.sample()
            rec = vae.decode(lat).sample
        dt=time.time()-t0; peak = rss()-base
        print(f"  tiling={str(tiling):<5} {sz}px  latent={tuple(lat.shape)}  "
              f"peak RSS delta ~{max(peak,0):6.0f} MB (fp32 CPU)  -> fp16 GPU ~{max(peak,0)/2:5.0f} MB   [{dt:.1f}s]")
        gc.collect()

print("\n" + "="*92)
print("EXP6b  training throughput.  i7-1355U CPU measured -> RTX 3050 Laptop 4GB (scaled)")
print("="*92)
import timm
from exp5_vram import SRNet
def bench(model, x, iters=3):
    opt = torch.optim.AdamW(model.parameters(), 1e-4); model.train()
    y = torch.randint(0,2,(x.shape[0],))
    for _ in range(1): # warmup
        opt.zero_grad(); torch.nn.functional.cross_entropy(model(x),y).backward(); opt.step()
    t0=time.time()
    for _ in range(iters):
        opt.zero_grad(); torch.nn.functional.cross_entropy(model(x),y).backward(); opt.step()
    return (time.time()-t0)/iters

# 3050 Laptop (GA107, 2048 CUDA cores, ~5.5 TFLOP fp32 / ~18 TFLOP fp16 tensor, 192 GB/s)
# vs i7-1355U AVX2 (~0.35 TFLOP fp32 effective).  Conservative measured-world speedup for
# small conv nets with AMP: ~18-25x.  Use 20x.
SPEEDUP = 20
cases = [("SRNet 256px bs=8", SRNet(), torch.randn(8,1,256,256)),
         ("EffNet-B0 256px bs=16", timm.create_model("efficientnet_b0",num_classes=2,in_chans=1), torch.randn(16,1,256,256)),
         ("EffNet-B0 512px bs=8",  timm.create_model("efficientnet_b0",num_classes=2,in_chans=1), torch.randn(8,1,512,512))]
print(f"{'config':<26}{'CPU s/it':>10}{'3050 s/it':>12}{'img/s':>9}   time for 1 epoch over 50k pairs")
print("-"*92)
for name, m, x in cases:
    t = bench(m, x); g = t/SPEEDUP; ips = x.shape[0]/g
    ep = 100000/ips/3600     # 50k covers + 50k stegos = 100k images per epoch
    print(f"{name:<26}{t:>10.2f}{g:>12.3f}{ips:>9.0f}   {ep:>5.2f} h/epoch")
    del m, x; gc.collect()
