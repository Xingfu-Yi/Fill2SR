# Inference Guide

See the [README](../README.md) for installation, model downloads and a quick example.

## Model Checkpoint

The released `sr-full.safetensors` checkpoint corresponds to **Ours-Full**, the paper's model with RCDT. Ours-Base weights, training code and RCDT data generation code are not included in this release.

The expected SHA-256 of `sr-full.safetensors` is:

```text
713a7c8e6cf6ca4265b0d699ecaff1dc0d13039b6ac171431dc68c3af695fbe6
```

Verify it with `sha256sum models/sr-full.safetensors` on Linux or `shasum -a 256 models/sr-full.safetensors` on macOS. Hugging Face downloads require authentication and access to the corresponding repository. A message falling back to an existing local file does not confirm a successful download.

### Model Paths

The scripts read `configs/config.yaml`. Relative model paths are resolved from the repository root, regardless of the directory from which a script is launched. Override them with `--fill-dir`, `--redux-dir` and `--lora` as needed. The `models/` directory is ignored by Git.

## Output Size

Choose either `--scale` or `--resolution`. The command-line default is **2×**; the Gradio demo defaults to the **2K** preset.

- **Scale:** output width and height equal the input dimensions multiplied by the requested scale.
- **Resolution:** output dimensions follow the input aspect ratio and the preset's pixel budget, with 16-pixel alignment. The preset name does not specify a fixed longest edge.

| Preset | Sizing convention | Output for a 256 × 256 input |
| --- | --- | --- |
| 2K | Original model preset | 1,536 × 1,536 |
| 4K | 3840 × 2160 pixel budget | 2,880 × 2,880 |
| 5K | 5120 × 2880 pixel budget | 3,840 × 3,840 |
| 8K | 7680 × 4320 pixel budget | 5,760 × 5,760 |
| 12K | 12288 × 6480 pixel budget | 8,928 × 8,928 |

Large outputs are restored over the complete target canvas using overlapping tiles.

For 4× super-resolution:

```bash
python scripts/infer.py \
  --input assets/examples/reallq250/113.png \
  --output outputs/nokia_4x.png \
  --scale 4 --steps 28 --seed 2026 --offload
```

## GPU Selection

`--offload` reduces GPU memory use at the cost of speed. Add `--device cuda:2` to select a GPU; both Fill and Redux use the selected device, including with CPU offload. Device indices are relative to `CUDA_VISIBLE_DEVICES`. Invalid or unavailable devices fail before checkpoint loading.

## Gradio Demo

The demo includes 16 original RealLQ250 inputs from appendix Figs. A3–A6, with Nokia first. Choose the **2K** resolution preset or **2× / 3× / 4×** scale, then select **Generate**. Sampling steps and seed are available under **Advanced settings**. Output dimensions, runtime and seed are shown after generation.

The [project website](https://xingfu-yi.github.io/Fill2SR/) displays precomputed results; run Gradio locally to generate new results. Website preview instructions are in [docs/README.md](README.md).

The interactive showcase uses 2K, 28 steps and seed 2026. The paper figures retain their original experimental settings, shown in each figure. The [showcase manifest](gallery.json) records image IDs, hashes, dimensions, model version and sampling parameters. These are generated restorations; fine details and text may differ from the original scene.

## Batch Inference

The batch entry point recursively reads images and writes PNG predictions while preserving relative directories. It generates predictions only; it does not download benchmark datasets or compute metrics.

Preview the input/output mapping without loading a model:

```bash
python scripts/evaluate.py \
  --input-dir path/to/LQ \
  --output-dir outputs/benchmark \
  --scale 2 --dry-run
```

Use the same settings as the interactive showcase:

```bash
python scripts/evaluate.py \
  --input-dir path/to/LQ \
  --output-dir outputs/showcase \
  --resolution 2k --steps 28 --seed 2026 --offload
```

Hidden files and directories, including macOS `._` sidecars, are skipped. Unreadable images are reported by filename before model loading. If multiple inputs map to the same PNG output, such as `photo.jpg` and `photo.png`, the command lists the conflicts and stops before model loading or output writes. Rename conflicting inputs or place them in different subdirectories. `--dry-run` performs the same checks.

## Debug Logging

When troubleshooting through the Python API, pass `debug=True` to `Fill2SRPipeline` to show resize, kernel and tiling details. Progress indicators, saved output paths and actionable warnings remain visible by default.
