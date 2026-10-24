"""Build a 512x512 test corpus from real photographs (never JPEG-recompressed by us)."""
import os, glob
from PIL import Image
import numpy as np

SRC = "/home/lap-030/Downloads"
OUT = "/home/lap-030/stego/rnd/corpus"
# Real camera/photo sources: prefer large natural photos over AI-generated PNGs,
# but include a couple of synthetic ones to see if texture statistics differ.
NAMES = [
    "japan-artistic-5120x2880-25406.jpg",
    "alone-sunset-silhouette-mountains-lake-reflections-starry-6579x4724-9067.jpg",
    "dolomites-kronplatz-5120x3200-11544.jpg",
    "20260808_071411241_iOS.jpg",
    "20260808_124527.jpg.jpeg",
    "81okge7rea7g1.png",
]

def tiles_from(path, n=4, size=512):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w < size or h < size:
        return []
    out = []
    # sample tiles spread across the image, skip flat ones (low variance = unrealistic)
    rng = np.random.default_rng(abs(hash(os.path.basename(path))) % (2**31))
    tries = 0
    while len(out) < n and tries < 200:
        tries += 1
        x = int(rng.integers(0, w - size)); y = int(rng.integers(0, h - size))
        t = im.crop((x, y, x + size, y + size))
        a = np.asarray(t.convert("L"), dtype=np.float64)
        if a.std() < 18:      # reject near-flat sky/black tiles
            continue
        out.append(t)
    return out

if __name__ == "__main__":
    k = 0
    for nm in NAMES:
        p = os.path.join(SRC, nm)
        if not os.path.exists(p):
            print("MISS", nm); continue
        for t in tiles_from(p):
            t.save(os.path.join(OUT, f"img{k:03d}.png"), compress_level=1)
            k += 1
    print(f"wrote {k} tiles of 512x512 -> {OUT}")
    # report texture stats
    for f in sorted(glob.glob(OUT + "/*.png"))[:100]:
        a = np.asarray(Image.open(f).convert("L"), float)
        print(f"  {os.path.basename(f)}  std={a.std():6.2f}  mean={a.mean():6.1f}")
