+++ 
date = '2025-07-14' 
title = 'Running an old AMD Datacenter card at home' 
draft = false 
+++

And why _you_ likely shouldn't

## Intro

I've been craving more parallel compute power in my home recently, and have been looking around for
a long time on something that can fit my student budget/work with my present hardware. Specifically
i had a old motherboard and 8th-gen Intel CPU laying around that i wanted to make use of. A few
alternatives i considered:

- The boring standard answer of buying a used RTX 3090 which seems to be still the best tradeoff on
  price/power/flops/memory for most people. But buying a 600€ card for a 100€ platform felt like a
  bit of an overkill, especially since my windows PC that i use for mostly gaming is only running a
  old GTX 1660.

- AMD and Intel have been putting out some pretty interesting _new_ GPUs like the 9070/9060 or the
  Arc B580s, and likely this would've been the more mature and easy solution, but something that
  irked me about this specifically the memory-bandwith to price ratio, since i was planing to run
  LLMs on it.

- The last one that im still considering, once my budget allows it, are the Tenstorrent cards
  (Wormhole/Blackhole), since they seem to not only be technicalogically interesting but also have
  some good power-efficiency from what i hear. But 1k€ is just too much for me currently, so this
  sadly has to wait (unless someone has a Grayskull card laying around that they want to get rid of).

---

So instead of choosing any of these sane, well-prove solutions, i decided to buy myself some problems.
Years ago i noticed that old NVIDIA Tesla HPC cards (V100 etc.) are going for quite cheap on eBay, and i got
reminded of that now recently when i saw someone on r/localllama talking about using a bunch of AMD Instinct Mi25 
for at home inference (also this [L1 forum post](https://forum.level1techs.com/t/mi25-stable-diffusions-100-hidden-beast/194172)).

So i looked on eBay and found a seller right here in Germany selling Refurbed Mi50s for ~110€ a piece, which on one evening 
around mightnight (best time for decisions obviously) decided to order, alongside an improved PSU since the card is sometimes
reported to have a power-limit of 300 watts. 3 days later and i received a beautiful, outdated, and shiny new toy:

{{< figure src="card.jpeg" alt="Image of the card" width="70%" >}}

Some interesting things to note about these cards:

- It sports 16GB of HBM2 memory at a reported ~ 1TB/s of bandwith  (which for that price has to be a contender for cheapest GB/s/€ ratio)
- 60 Compute-units (CUs) AMDs equivalent of the SMs with a wavefront/warpsize of 64 (*not the usual 32*), though ive read somewhere that this is configurable
- only ~27 TFLOP/s of FP16 😿 (and afaik no support for any smaller datatypes, so no Q4/Q8 quantizations)

So while its definitely an interesting card, there's a reason they're so cheap to get... primarily that horrendous TFLOPS/W ratio. 

## Setup problems

I've had this card for close to a week now and had quite a few hurdles to cross until i got it into a somewhat operating state.

- As mentioned i had to buy a PSU that supported the 300Watt via 2 8-pin PCI cables, but that wouldve likely been a given with any of the alternatives
- The card is built for a blowthrough-style server chasis, so i had to MacGyver myself a fan shroud/duct on the end of it

{{< figure src="installed.jpeg" alt="Image of the system with the pink 3D-printed fan-duct" width="70%" >}}
I still haven't  fully load tested the card, so im unsure if this arrangment will be enough to cool it. But sofar its holding the GPU at <45 celsius.

I thought with this i might have halve of the work done, but it turned out the hardware side was alot more straightforward to solve, compared to the 
hustle the software turned out to be (and still is). For documentations sake (if anyone else should be naive enough to try and set these up) i will quickly
detail my issues and how i solved/circumvented them:

1. While some docs talk of howto setup the drivers for the card on Arch, i mostly found Ubuntu related docs, specifically Ubuntu 22.04 LTS. I initially tried 24.04, but ditched it since
 the offical docs for the ROCm version i chose only talk about Ubuntu 20.04 and 22.04.
2. Enable 'Above 4G decoding' in the BIOS settings, which as-far as i understand it enables PCIe devices to use larger then 32-bit virtual addresses or something, but im not quite sure. This is definitely required for the card to run though!
3. I had to twiddle around with my linux-kernel cmdline arguments quite a bit, the final solution that worked for me is `pci_realloc pci=realloc,hpmemsize=512M,hpiosize=32M` but dont ask me to explain this to you, as i dont fully understand them either.
4. The Vega 20/GFX-9 ISA got dropped from the offical ROCm support list, so i chose a older ROCm release for my setup (5.6.0) as the newest one i tried (6.4.0) didn't end up working

After all this the card finally got recognized by the amdgpu-drivers, but while trying to build the HIP-Basics from [ROCm-examples](https://github.com/ROCm/rocm-examples) i also had to install
the libstdc++-12-dev package to fix the `fatal error: 'cmath' file not found` linker errors. Finally one should also not forget to add their own user to the `render` group, since otherwise the 
user-space programs (like `rocm-smi`) aren't allowed to interface with the card and dont report it (this is totally not an issue that bugged me for an hour...).

## Results.. ?

But with all this done i can finally run some code on it, in this case ive mostly focused on getting the HIP-basics running and then some of my tinygrad based ML projects.

For example the basic mat-mul kernel from the HIP-Basics:

```fish
jakob@jupiter ~/g/r/H/matrix_multiplication> make && ./hip_matrix_multiplication
/opt/rocm/bin/hipcc -std=c++17 -Wall -Wextra -I ../../Common -I ../../External  -o hip_matrix_multiplication main.hip
starting calculation
Matrix multiplication: [8192x8192] * [8192x8192], block size: 16x16
Duration: 472.72 ms
GFLOPs: 2325.95
```

_NOTE: The mat-mul kernel is relatively basic and not optimized, with a more tuned kernel i would expect almost 10x of that FLOPs number._

Or i can run the tinygrad examples:
```fish
jakob@jupiter ~/g/t/examples (master)> GPU=1 uv run beautiful_mnist.py
loss:   0.09 test_accuracy: 98.19%: 100%|█████████████████████| 70/70 [00:09<00:00,  7.01it/s]
```

This uses the OpenCL backend if im not quite wrong, i know tinygrad has a RX7900 specific backend or something close to that, but for now i think we are going via OpenCL.

I might write another blogpost about the results of me learning HIP and understanding the architecture a little better, but for now im relatively happy i got it all working at all.


##  tl;dr

Should you buy yourself an Instinct Mi50 (or any of these older HPC cards for that matter)? 

Likely not (with a caveat). 

A definite no-no if you want to make money with it, since the power usage almost immediatly invalidates its existence, especially with the power-costs here in Germany.
It's also not super helpful if you want to just use LLMs since i have yet to find an inference-tool that works on the GPU, and since it doesn't support anything smaller then FP16 you
likely wont find a good application there for it either.

I would say really the only category of people that should buy this are people that are already familar with GPUs (but maybe not AMD GPUs) or just want to tinker around a little with out 
throwing out too much money immediatley. It's faster for ML-training then my M2 Macbook Air and also then my GTX 1660 via WSL, so thats something. But i wont likely leave it running over night,
especially since if im desperate i can still use my uni's HPC-cluster.
