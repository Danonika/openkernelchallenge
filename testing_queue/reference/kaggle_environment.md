# Verified Kaggle environment

Compiled 30 October 2026, 18:24 UTC. Hardware was observed live at 16:40 UTC; software/driver were confirmed in actual runs through 16:45 UTC. No current live kernel probe was established. No new session, workload or download was started by this package. Current kernel state is unknown.

| Item | Verified value |
|---|---|
| NVIDIA driver | **580.178.04** |
| Torch build / compiled CUDA | **2.11.0+cu128 / 12.8** |
| nvidia-smi advertised CUDA maximum | **Not captured; unknown** |
| GPU |2× Tesla T4 exposed; **1 used**, CUDA_VISIBLE_DEVICES=0 |
| VRAM |15,360 MiB per GPU (15 GiB reported) |
| Compute capability |7.5 / SM75 |
| CPU |Intel(R) Xeon(R) CPU @ 2.00GHz |
| Usable CPUs |4 logical; affinity 4; quota 400000/100000=4; 2 physical reported |
| RAM limit |30 GiB cgroup limit; host-visible total≈31.35 GiB |
| Python |3.13.15 |
| Triton |3.6.0 |
| Soup CLI |0.77.0 |
| Transformers |5.16.1 |
| PEFT |0.20.0 |
| TRL |0.29.1 |
| bitsandbytes |0.50.2 |
| Accelerate |1.14.0 |
| datasets |4.8.5 |
| pytest / full pip freeze |**Not captured; unknown** |
| OS distribution / kernel |**Not recorded; unknown** |
| Allocator |expandable_segments:True |

The CUDA version shown by `nvidia-smi` is the driver’s maximum supported CUDA version—not the CUDA version Torch was built with. Here, **Torch 12.8 is proven; the nvidia-smi header value is not**. It must not be guessed from the driver series. [NVIDIA documentation](https://docs.nvidia.com/deploy/nvidia-smi/)

`kaggle_environment.json` contains field-level historical/live markers and exact artifact SHA256s. This is a portable deployment reference, not proof of identical OS image or timing on another provider. A future approved read-only live probe could fill the unknown fields without training or downloading anything.
