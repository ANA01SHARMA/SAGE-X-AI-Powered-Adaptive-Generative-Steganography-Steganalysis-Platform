"""EXP2: how detectable is each embedding family? SPAM-686 features + linear classifier.
This is a WEAK detector on purpose - it is the floor. A CNN (SRNet/EffNet) does strictly better.
If a method is detectable by SPAM, it is hopeless against a real steganalyzer."""
import glob, numpy as np, time
from PIL import Image
import stegolib as S

T = 3
def spam686(x):
    x = x.astype(np.int16); f = []
    for D in ((0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)):
        d = np.roll(x, D, (0,1)) - x
        d = np.clip(d, -T, T) + T
        a, b, c = d[:-2,:-2].ravel(), d[1:-1,1:-1].ravel(), d[2:,2:].ravel()
        h = np.bincount(a*(2*T+1)**2 + b*(2*T+1) + c, minlength=(2*T+1)**3).astype(np.float64)
        f.append(h/max(h.sum(),1))
    f = np.array(f)
    return np.concatenate([f[:4].mean(0), f[4:].mean(0)])   # 686-D

def fld_ensemble(Xc, Xs, n_sub=40, d_sub=180, seed=0):
    """Kodovsky's ensemble of FLDs - the standard steganalysis classifier."""
    rng = np.random.default_rng(seed)
    n = len(Xc); tr = int(n*0.6)
    idx = rng.permutation(n); trn, tst = idx[:tr], idx[tr:]
    votes_c = np.zeros(len(tst)); votes_s = np.zeros(len(tst))
    for _ in range(n_sub):
        sub = rng.choice(Xc.shape[1], d_sub, replace=False)
        A, B = Xc[trn][:,sub], Xs[trn][:,sub]
        mA, mB = A.mean(0), B.mean(0)
        Sw = np.cov(A.T)+np.cov(B.T)+np.eye(d_sub)*1e-6
        try: w = np.linalg.solve(Sw, mB-mA)
        except np.linalg.LinAlgError: continue
        thr = ((mA+mB)/2)@w
        votes_c += (Xc[tst][:,sub]@w > thr); votes_s += (Xs[tst][:,sub]@w > thr)
    fa = np.mean(votes_c > n_sub/2); md = np.mean(votes_s <= n_sub/2)
    return (fa+md)/2, len(tst)          # P_E (min average decision error)

FILES = sorted(glob.glob("/home/lap-030/stego/rnd/corpus256/*.png"))[:300]

def evaluate(name, fn, note=""):
    t0=time.time(); Xc, Xs = [], []
    for f in FILES:
        g = np.array(Image.open(f).convert("L"))
        s = fn(g)
        Xc.append(spam686(g)); Xs.append(spam686(s))
    pe, nt = fld_ensemble(np.array(Xc), np.array(Xs))
    sec = "SECURE(vs SPAM)" if pe>0.40 else ("weak" if pe>0.25 else "BROKEN - trivially detected")
    print(f"{name:<44} P_E={pe:.3f}  acc={1-pe:.3f}   {sec}  {note} [{time.time()-t0:.0f}s]")
    return pe

if __name__ == "__main__":
    N = 256*256
    print("="*96)
    print(f"EXP2  detectability, {len(FILES)} cover/stego pairs @256x256, SPAM-686 + FLD ensemble")
    print("P_E = min average decision error.  0.50 = undetectable,  0.00 = perfectly detected")
    print("="*96)

    evaluate("LSB-matching  0.40 bpp",  lambda g: S.embed_lsbm(g, S.bits(int(N*0.40))), "(26214 bits)")
    evaluate("LSB-matching  0.10 bpp",  lambda g: S.embed_lsbm(g, S.bits(int(N*0.10))), "(6553 bits)")
    evaluate("LSB-matching  0.01 bpp",  lambda g: S.embed_lsbm(g, S.bits(int(N*0.01))), "(655 bits)")
    evaluate("HILL adaptive 0.40 bpp",  lambda g: S.embed_hill_sim(g, int(N*0.40))[0],  "(26214 bits)")
    evaluate("HILL adaptive 0.10 bpp",  lambda g: S.embed_hill_sim(g, int(N*0.10))[0],  "(6553 bits)")
    evaluate("HILL adaptive 0.01 bpp",  lambda g: S.embed_hill_sim(g, int(N*0.01))[0],  "(655 bits)")
    print("-"*96)
    print("ROBUST methods - the ones that actually survive a channel:")
    evaluate("robust QIM B=32 d=18  (64 payload bits)",
             lambda g: S.embed_rqim(g, S.bits(16), 32, 18.0, 4)[0], "(rep=4)")
    evaluate("robust QIM B=64 d=40  (16 payload bits)",
             lambda g: S.embed_rqim(g, S.bits(4), 64, 40.0, 4)[0], "(rep=4)")
    evaluate("robust QIM B=32 d=6   (weakened delta)",
             lambda g: S.embed_rqim(g, S.bits(16), 32, 6.0, 4)[0], "(rep=4)")
