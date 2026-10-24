"""Larger 256x256 corpus for steganalysis training."""
import os, glob, numpy as np
from PIL import Image
SRC="/home/lap-030/Downloads"; OUT="/home/lap-030/stego/rnd/corpus256"
os.makedirs(OUT, exist_ok=True)
NAMES=["japan-artistic-5120x2880-25406.jpg",
 "alone-sunset-silhouette-mountains-lake-reflections-starry-6579x4724-9067.jpg",
 "dolomites-kronplatz-5120x3200-11544.jpg","20260808_071411241_iOS.jpg",
 "20260808_124527.jpg.jpeg","81okge7rea7g1.png","Image.jpeg",
 "image_2026-09-17T14-48-55-347Z.jpg"]
k=0
for nm in NAMES:
    p=os.path.join(SRC,nm)
    if not os.path.exists(p): continue
    im=Image.open(p).convert("L"); w,h=im.size
    if w<256 or h<256: continue
    # dense non-overlapping grid, keep textured tiles
    for y in range(0,h-256,256):
        for x in range(0,w-256,256):
            t=im.crop((x,y,x+256,y+256)); a=np.asarray(t,float)
            if a.std()<15: continue
            t.save(f"{OUT}/t{k:04d}.png",compress_level=1); k+=1
            if k>=500: break
        if k>=500: break
    if k>=500: break
print("tiles:",k)
