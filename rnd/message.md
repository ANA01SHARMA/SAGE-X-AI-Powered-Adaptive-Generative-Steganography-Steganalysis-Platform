Here's where we landed on SAGE-X after actually testing things instead of going by papers.

**Three things we measured that changed the plan**

1. Hiding data in pixels doesn't survive compression at all. We got a bit error rate of 0.49
   after JPEG at quality 95 — that's a coin flip, i.e. nothing recovers. Not "degraded",
   completely gone. Same even for tiny payloads.

2. You can't have robust AND undetectable at the same time. We swept the embedding strength
   and the curve turns out to be a step — the weakest setting that survives compression is
   already the first setting a basic detector catches. There's no middle ground, so the
   "adaptive feedback loop" from our proposal has nothing to actually tune.

3. The generative/Stable-Diffusion part isn't worth doing even with a bigger GPU. Recent work
   hits the same wall we did — pixel-space models are secure but fragile, latent models are
   robust but detectable. On top of that, decoding from latents needs sender and receiver to
   reproduce identical floating-point outputs, which breaks across different GPUs and library
   versions, and the receiver would need a multi-GB model. Not worth the risk. (Our 4GB GPU
   also can't train it, but that's the smaller problem.)

**What we're building instead**

A complete working system, end to end:

  encrypt a file or message -> hide it inside a cover image -> share it -> receiver extracts
  and decrypts it, byte-for-byte identical

Plus three things on top:

- **Capacity check before embedding.** It works out whether the data can actually be
  recovered *first*, and refuses if it can't.
- **Splitting across multiple images.** A big file spread over several photos with erasure
  coding — any 4 out of 6 images recover the whole thing. You can delete two and it still works.
- **Public-key mode.** Sender and receiver don't need to agree on a password beforehand.

**The one thing we can claim over existing tools**

Every tool out there — Steghide, OpenStego, SilentEye — hides whatever you give it whether or
not it can come back. You only find out it failed when extraction returns nothing.

Concrete case we measured: a 512x512 image sent through a compressing channel can't carry even
the 34-byte encryption header. Those tools will still embed into it and report success.

Ours checks first and tells you what will actually fit. We haven't found another tool that does.

**What we're dropping**

Generative image synthesis, latent-space hiding, and robustness to compression/cropping — for
the reasons above. Steganalysis stays (it fits our GPU fine at 1.2GB) and we use it to set the
safe embedding rate from measurement instead of just picking a number.

All the measurements and scripts are saved and re-runnable if anyone wants to see them.
