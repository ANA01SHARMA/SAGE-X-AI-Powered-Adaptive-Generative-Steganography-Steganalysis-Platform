"""EXP1: the capacity-vs-robustness frontier. How much survives a real-world channel?"""
import glob, numpy as np, time
from PIL import Image
import stegolib as S

CORPUS = sorted(glob.glob("/home/lap-030/stego/rnd/corpus/*.png"))
ATTACKS = [
    ("clean",        lambda x: S.a_none(x)),
    ("JPEG q95",     lambda x: S.a_jpeg(x, 95)),
    ("JPEG q90",     lambda x: S.a_jpeg(x, 90)),
    ("JPEG q75",     lambda x: S.a_jpeg(x, 75)),
    ("JPEG q50",     lambda x: S.a_jpeg(x, 50)),
    ("WebP q90",     lambda x: S.a_webp(x, 90)),
    ("WebP q80",     lambda x: S.a_webp(x, 80)),
    ("resize .75",   lambda x: S.a_resize(x, 0.75)),
    ("resize .50",   lambda x: S.a_resize(x, 0.50)),
    ("crop 90%",     lambda x: S.a_crop(x, 0.90)),
    ("blur s=0.5",   lambda x: S.a_blur(x, 0.5)),
    ("noise s=2",    lambda x: S.a_noise(x, 2.0)),
]

def run(name, embed, extract, npayload, imgs, extra=""):
    rows = {a: [] for a, _ in ATTACKS}
    ps = []
    for f in imgs:
        g = np.array(Image.open(f).convert("L"))
        p = S.bits(npayload, seed=hash(f) % 1000)
        st, key = embed(g, p)
        ps.append(S.psnr(g, st))
        for an, af in ATTACKS:
            try:
                rows[an].append(S.ber(p, extract(af(st), key, npayload)))
            except Exception:
                rows[an].append(1.0)
    print(f"\n### {name}   payload={npayload} bits ({npayload/(512*512):.4f} bpp)  "
          f"PSNR={np.mean(ps):.1f}dB {extra}")
    print(f"{'attack':<14}{'BER':>8}   {'verdict'}")
    for an, _ in ATTACKS:
        b = float(np.mean(rows[an]))
        v = "RECOVERABLE" if b < 0.02 else ("ECC-able" if b < 0.12 else "DESTROYED")
        print(f"{an:<14}{b:>8.4f}   {v}")
    return {a: float(np.mean(v)) for a, v in rows.items()}

if __name__ == "__main__":
    imgs = CORPUS[:8]
    t0 = time.time()
    print("="*62); print("EXP1  robustness of each embedding family  (n=%d images, 512x512 gray)" % len(imgs))
    print("="*62)

    # --- A: spatial LSB matching, classic steganography payload
    run("A. LSB-matching  (spatial, the usual 'AI stego' starting point)",
        lambda g, p: (S.embed_lsbm(g, p), None),
        lambda s, k, n: S.extract_lsbm(s, n),
        104857, imgs)

    # --- A': same method, tiny payload — does less payload help robustness?
    run("A'. LSB-matching at a tiny payload (256 bits)",
        lambda g, p: (S.embed_lsbm(g, p), None),
        lambda s, k, n: S.extract_lsbm(s, n),
        256, imgs)

    # --- C: JPEG-DCT domain
    def emb_c(g, p):
        st, n, cap, key = S.embed_jdct(g, p, qf=90)
        return st, key
    run("C. JPEG-DCT LSB, QF90-locked (F5-style)",
        emb_c, lambda s, k, n: S.extract_jdct(s, k, n),
        4096, imgs)

    # --- D: robust block QIM, several redundancy settings
    for B, delta, rep in [(32, 18.0, 1), (32, 18.0, 4), (64, 26.0, 4), (64, 40.0, 4)]:
        nb = (512 // B) ** 2
        npay = max(8, nb // rep)
        run(f"D. Robust block-DCT QIM  B={B} delta={delta} rep={rep}",
            lambda g, p, B=B, delta=delta, rep=rep: (S.embed_rqim(g, p, B, delta, rep)[0],
                                                     (B, delta, rep)),
            lambda s, k, n: S.extract_rqim(s, n, k[0], k[1], k[2]),
            npay, imgs, extra=f"(capacity={nb} blocks)")
    print(f"\n[total {time.time()-t0:.0f}s]")
