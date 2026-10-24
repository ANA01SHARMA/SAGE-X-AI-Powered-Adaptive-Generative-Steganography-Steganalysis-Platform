"""EXP4: is crop recoverable? Crop desynchronises the block grid - that is a SYNC failure,
not an information failure. Test whether a scale+offset search resynchronises the decoder."""
import glob, numpy as np, time
from PIL import Image
import stegolib as S

B, DELTA, REP, NPAY = 32, 26.0, 4, 64
PRE = np.array([1,0,1,1,0,0,1,0,1,1,1,0,0,1,0,1], np.uint8)   # 16-bit sync preamble

def embed_with_preamble(g, payload):
    full = np.concatenate([PRE, payload])
    st, k, nb = S.embed_rqim(g, full, B, DELTA, REP)
    return st

def extract_at(img, scale, dx, dy, n):
    im = Image.fromarray(img)
    w, h = im.size
    im = im.resize((max(64,int(w*scale)), max(64,int(h*scale))), Image.BICUBIC)
    a = np.array(im)
    a = a[dy:, dx:]
    if a.shape[0] < B*4 or a.shape[1] < B*4: return None
    return S.extract_rqim(a, n, B, DELTA, REP)

def resync(img, n):
    """search scale x offset, score by preamble match"""
    best = (-1, None, None)
    for scale in np.arange(0.86, 1.22, 0.02):
        for dy in range(0, B, 4):
            for dx in range(0, B, 4):
                bits = extract_at(img, scale, dx, dy, len(PRE))
                if bits is None: continue
                score = int((bits[:len(PRE)] == PRE).sum())
                if score > best[0]: best = (score, scale, (dx, dy))
    return best

if __name__ == "__main__":
    files = sorted(glob.glob("/home/lap-030/stego/rnd/corpus/*.png"))[:4]
    print("="*88)
    print(f"EXP4  crop resynchronisation  (B={B}, delta={DELTA}, rep={REP}, 16-bit preamble + {NPAY} bits)")
    print("="*88)
    for cropfrac in [1.0, 0.95, 0.90, 0.80]:
        naive, synced, found = [], [], []
        t0 = time.time()
        for f in files:
            g = np.array(Image.open(f).convert("L"))
            p = S.bits(NPAY, seed=hash(f) % 999)
            st = embed_with_preamble(g, p)
            att = st if cropfrac == 1.0 else S.a_crop(st, cropfrac)
            # naive decode, assume nothing happened
            nb = S.extract_rqim(att, len(PRE)+NPAY, B, DELTA, REP)
            naive.append(S.ber(p, nb[len(PRE):]))
            # resync decode
            score, scale, off = resync(att, len(PRE)+NPAY)
            found.append((score, round(float(scale),2), off))
            if scale is None: synced.append(1.0); continue
            sb = extract_at(att, scale, off[0], off[1], len(PRE)+NPAY)
            synced.append(S.ber(p, sb[len(PRE):]) if sb is not None else 1.0)
        tag = "no crop" if cropfrac==1.0 else f"crop {int(cropfrac*100)}% + rescale"
        print(f"{tag:<26} naive BER={np.mean(naive):.3f}   after resync BER={np.mean(synced):.3f}"
              f"   preamble {np.mean([f[0] for f in found]):.1f}/16  scales={[f[1] for f in found]}"
              f"  [{time.time()-t0:.0f}s]")
