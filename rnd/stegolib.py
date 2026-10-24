"""Embedding families + attack suite. Pure numpy/PIL/scipy — no GPU needed."""
import io, numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from scipy.ndimage import convolve, uniform_filter, gaussian_filter

# ---------------------------------------------------------------- utilities
def bits(n, seed=7):
    return np.random.default_rng(seed).integers(0, 2, n).astype(np.uint8)

def ber(a, b):
    n = min(len(a), len(b))
    return float(np.mean(a[:n] != b[:n])) if n else 1.0

def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m == 0 else 10 * np.log10(255.0 ** 2 / m)

def dct2(b):  return dct(dct(b, axis=0, norm='ortho'), axis=1, norm='ortho')
def idct2(b): return idct(idct(b, axis=0, norm='ortho'), axis=1, norm='ortho')

def blockify(a, B):
    h, w = a.shape; h -= h % B; w -= w % B
    return (a[:h, :w].reshape(h // B, B, w // B, B).swapaxes(1, 2)), h, w

def unblockify(blk, h, w, B):
    return blk.swapaxes(1, 2).reshape(h, w)

# ---------------------------------------------------------------- A: LSB matching
def embed_lsbm(cover, payload, seed=1):
    x = cover.copy().astype(np.int16)
    idx = np.random.default_rng(seed).permutation(x.size)[:len(payload)]
    flat = x.ravel()
    cur = flat[idx] & 1
    need = cur != payload
    step = np.random.default_rng(seed + 1).choice([-1, 1], size=len(payload))
    v = flat[idx] + np.where(need, step, 0)
    v = np.where(v < 0, 1, np.where(v > 255, 254, v))
    flat[idx] = v
    return x.astype(np.uint8).reshape(cover.shape)

def extract_lsbm(stego, n, seed=1):
    idx = np.random.default_rng(seed).permutation(stego.size)[:n]
    return (stego.ravel()[idx] & 1).astype(np.uint8)

# ---------------------------------------------------------------- B: HILL adaptive +-1
KB = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], float)

def hill_cost(cover):
    r = convolve(cover.astype(float), KB, mode='mirror')
    c = uniform_filter(np.abs(r), 3, mode='mirror')
    c = 1.0 / np.maximum(c, 1e-8)
    c = uniform_filter(c, 15, mode='mirror')
    return np.minimum(c, 1e8)

def _lambda_for(rho, m):
    """binary-search lambda so ternary entropy of embedding probs == m bits"""
    def H(lam):
        p = 1.0 / (1.0 + 2.0 * np.exp(-lam * rho))       # p(stay)
        q = (1.0 - p) / 2.0                               # p(+1)=p(-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            h = -(p * np.log2(p) + 2 * q * np.log2(np.maximum(q, 1e-300)))
        return np.nansum(h)
    lo, hi = 1e-6, 1e6
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if H(mid) > m: lo = mid
        else: hi = mid
    return np.sqrt(lo * hi)

def embed_hill_sim(cover, nbits, seed=2):
    """Optimal-simulator embedding at payload nbits (what STC coding would achieve)."""
    rho = hill_cost(cover)
    lam = _lambda_for(rho, nbits)
    p_stay = 1.0 / (1.0 + 2.0 * np.exp(-lam * rho))
    q = (1.0 - p_stay) / 2.0
    rng = np.random.default_rng(seed)
    u = rng.random(cover.shape)
    d = np.where(u < q, -1, np.where(u < 2 * q, 1, 0))
    s = cover.astype(np.int16) + d
    s = np.clip(s, 0, 255)
    return s.astype(np.uint8), int((s != cover).sum())

# ---------------------------------------------------------------- C: JPEG-DCT LSB (QF-locked)
STD_Q = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
                  [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
                  [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
                  [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]], float)

def qtable(qf):
    s = 5000 / qf if qf < 50 else 200 - 2 * qf
    q = np.floor((STD_Q * s + 50) / 100)
    return np.clip(q, 1, 255)

def _jpeg_coeffs(gray, Q):
    blk, h, w = blockify(gray.astype(float) - 128.0, 8)
    C = np.zeros_like(blk)
    for i in range(blk.shape[0]):
        for j in range(blk.shape[1]):
            C[i, j] = np.round(dct2(blk[i, j]) / Q)
    return C, h, w

def _coeffs_to_pixels(C, Q, h, w):
    out = np.zeros((C.shape[0], C.shape[1], 8, 8))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            out[i, j] = idct2(C[i, j] * Q)
    return np.clip(unblockify(out, h, w, 8) + 128.0, 0, 255)

def embed_jdct(gray, payload, qf=90, seed=3):
    """LSB of nonzero AC coefficients in the quantized DCT domain, locked to QF."""
    Q = qtable(qf)
    C, h, w = _jpeg_coeffs(gray, Q)
    flat = C.reshape(-1, 64)
    ac = np.zeros_like(flat, bool); ac[:, 1:] = True
    usable = ac & (flat != 0) & (np.abs(flat) > 0)
    pos = np.argwhere(usable)
    rng = np.random.default_rng(seed)
    pos = pos[rng.permutation(len(pos))]
    n = min(len(payload), len(pos))
    for k in range(n):
        b, c = pos[k]
        v = flat[b, c]
        if (int(abs(v)) & 1) != payload[k]:
            v = v - np.sign(v)            # move toward zero (F5-style)
            if v == 0: v = np.sign(flat[b, c]) * 2   # avoid creating a zero (unusable)
        flat[b, c] = v
    C2 = flat.reshape(C.shape)
    px = _coeffs_to_pixels(C2, Q, h, w)
    return np.round(px).astype(np.uint8), n, len(pos), (pos[:n], Q)

def extract_jdct(gray, key, n):
    pos, Q = key
    C, h, w = _jpeg_coeffs(gray, Q)
    flat = C.reshape(-1, 64)
    out = np.zeros(n, np.uint8)
    for k in range(n):
        b, c = pos[k]
        out[k] = int(abs(flat[b, c])) & 1
    return out

# ---------------------------------------------------------------- D: robust block-DCT QIM
def embed_rqim(gray, payload, B=32, delta=18.0, rep=1, seed=4, coef=(3, 1)):
    """Differential QIM on one mid-band DCT coefficient per BxB block, with repetition."""
    blk, h, w = blockify(gray.astype(float), B)
    nb = blk.shape[0] * blk.shape[1]
    stream = np.repeat(payload, rep)[:nb]
    out = blk.copy()
    u, v = coef
    k = 0
    for i in range(blk.shape[0]):
        for j in range(blk.shape[1]):
            if k >= len(stream): break
            D = dct2(blk[i, j])
            c = D[u, v]
            # quantize c to even/odd multiple of delta according to the bit
            q = np.round(c / delta)
            if (int(q) & 1) != stream[k]:
                q = q + (1 if c >= q * delta else -1)
            D[u, v] = q * delta
            out[i, j] = idct2(D)
            k += 1
    px = np.clip(unblockify(out, h, w, B), 0, 255)
    return np.round(px).astype(np.uint8), k, nb

def extract_rqim(gray, n, B=32, delta=18.0, rep=1, coef=(3, 1)):
    blk, h, w = blockify(gray.astype(float), B)
    u, v = coef
    raw = []
    for i in range(blk.shape[0]):
        for j in range(blk.shape[1]):
            D = dct2(blk[i, j])
            raw.append(int(np.round(D[u, v] / delta)) & 1)
    raw = np.array(raw[: n * rep], np.uint8)
    if rep == 1: return raw[:n]
    m = len(raw) // rep * rep
    votes = raw[:m].reshape(-1, rep).sum(1)
    return (votes > rep / 2).astype(np.uint8)[:n]

# ---------------------------------------------------------------- attacks
def a_jpeg(img, q):
    b = io.BytesIO(); Image.fromarray(img).save(b, 'JPEG', quality=q)
    return np.array(Image.open(io.BytesIO(b.getvalue())).convert('L'))

def a_jpeg_qt(img, Q):
    """re-save with an explicit quantisation table (QF-locked channel)"""
    b = io.BytesIO()
    Image.fromarray(img).save(b, 'JPEG', qtables=[Q.astype(int).flatten().tolist()])
    return np.array(Image.open(io.BytesIO(b.getvalue())).convert('L'))

def a_webp(img, q):
    b = io.BytesIO(); Image.fromarray(img).save(b, 'WEBP', quality=q)
    return np.array(Image.open(io.BytesIO(b.getvalue())).convert('L'))

def a_resize(img, s):
    im = Image.fromarray(img); w, h = im.size
    return np.array(im.resize((max(1,int(w*s)), max(1,int(h*s))), Image.BICUBIC)
                      .resize((w, h), Image.BICUBIC))

def a_crop(img, frac):
    im = Image.fromarray(img); w, h = im.size
    cw, ch = int(w*frac), int(h*frac)
    x, y = (w-cw)//2, (h-ch)//2
    return np.array(im.crop((x, y, x+cw, y+ch)).resize((w, h), Image.BICUBIC))

def a_blur(img, s):  return np.clip(gaussian_filter(img.astype(float), s), 0, 255).astype(np.uint8)
def a_noise(img, s):
    rng = np.random.default_rng(0)
    return np.clip(img.astype(float) + rng.normal(0, s, img.shape), 0, 255).astype(np.uint8)
def a_none(img): return img
