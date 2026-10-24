"""EXP3: the operating curve. Sweep embedding strength -> (robustness, detectability, quality).
This IS the search space the 'adaptive feedback loop' would operate over."""
import glob, numpy as np
from PIL import Image
import stegolib as S
from exp2_detectability import spam686, fld_ensemble

F256 = sorted(glob.glob("/home/lap-030/stego/rnd/corpus256/*.png"))[:240]
F512 = sorted(glob.glob("/home/lap-030/stego/rnd/corpus/*.png"))[:8]
CHAN = [("JPEG q90", lambda x: S.a_jpeg(x,90)), ("JPEG q75", lambda x: S.a_jpeg(x,75)),
        ("JPEG q50", lambda x: S.a_jpeg(x,50)), ("WebP q80", lambda x: S.a_webp(x,80)),
        ("resz .50", lambda x: S.a_resize(x,0.50)), ("noise2", lambda x: S.a_noise(x,2.0))]

print("="*104)
print("EXP3  robust block-DCT QIM, B=32, rep=4  ->  64 payload bits per 512x512 image")
print("      sweeping embedding strength delta.  P_E vs SPAM-686 (a WEAK detector: real floor is lower)")
print("="*104)
hdr = f"{'delta':>6}{'PSNR':>7}{'P_E':>7}{'det.acc':>9} | " + "".join(f"{c:>10}" for c,_ in CHAN)
print(hdr); print("-"*len(hdr))

for delta in [3.0, 6.0, 9.0, 12.0, 18.0, 26.0, 40.0]:
    # detectability @256 (B=16 keeps the same blocks-per-area ratio as B=32 @512)
    Xc, Xs = [], []
    for f in F256:
        g = np.array(Image.open(f).convert("L"))
        s = S.embed_rqim(g, S.bits(16), 16, delta, 4)[0]
        Xc.append(spam686(g)); Xs.append(spam686(s))
    pe,_ = fld_ensemble(np.array(Xc), np.array(Xs))
    # robustness + quality @512
    bers = {c: [] for c,_ in CHAN}; ps = []
    for f in F512:
        g = np.array(Image.open(f).convert("L")); p = S.bits(64, seed=hash(f)%999)
        st,_,_ = S.embed_rqim(g, p, 32, delta, 4)
        ps.append(S.psnr(g, st))
        for c, fn in CHAN:
            bers[c].append(S.ber(p, S.extract_rqim(fn(st), 64, 32, delta, 4)))
    print(f"{delta:>6.0f}{np.mean(ps):>7.1f}{pe:>7.3f}{1-pe:>9.3f} | " +
          "".join(f"{np.mean(bers[c]):>10.3f}" for c,_ in CHAN))

print("\nBER legend:  <0.02 clean recovery |  0.02-0.12 recoverable with ECC  |  >0.12 lost")
print("P_E legend:  0.50 undetectable | 0.40-0.50 marginal | <0.40 detected by even a weak detector")
