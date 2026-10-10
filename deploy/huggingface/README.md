---
title: Fill2SR
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.43.1
python_version: 3.12
app_file: deploy/huggingface/app.py
license: apache-2.0
short_description: ECCV 2026 · Real-world image super-resolution
models:
  - yixingfu/Fill2SR
  - black-forest-labs/FLUX.1-Fill-dev
  - black-forest-labs/FLUX.1-Redux-dev
tags:
  - image-to-image
  - super-resolution
  - research
---

# Fill2SR · ECCV 2026

**Repurposing Inpainting Diffusion Transformers for Real-World Super-Resolution**

Xingfu Yi · Xiaoxue Yu

[Paper](https://doi.org/10.1007/978-3-032-37556-8_24) · [Code](https://github.com/Xingfu-Yi/Fill2SR) · [Project Page](https://xingfu-yi.github.io/Fill2SR/) · [Weights](https://huggingface.co/yixingfu/Fill2SR)

Upload a low-quality image or choose an example, then select **Generate**. This demo uses the paper's **Ours-Full** checkpoint. Defaults: **2K, 28 steps, seed 2026**.

The 2K preset follows the input aspect ratio; a 256 × 256 square input produces a 1,536 × 1,536 result. Generated details and text may differ from the original scene.

## Use and attribution

This research demo uses FLUX.1-Fill-dev and FLUX.1-Redux-dev by Black Forest Labs. The Ours-Full weights are governed by the [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/yixingfu/Fill2SR/blob/main/LICENSE.md), including its non-commercial and non-production restrictions. The model repository contains the license and attribution. The demo code is Apache-2.0; Fill2SR is an independent research project.

RealLQ250 examples are from [DreamClear](https://github.com/shallowdream204/DreamClear); image rights remain with their respective owners.

## Running a copy

Choose ZeroGPU if your account supports it. Set an `HF_TOKEN` **Secret** with read access to the model repositories, after obtaining the required model access. Keep the Space private while testing. The deployment guide is [here](deploy/huggingface/DEPLOYMENT.md).
