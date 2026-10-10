# Hugging Face Spaces deployment

This directory prepares the ECCV 2026 **Ours-Full** Gradio demo for Hugging Face Spaces. It reuses `scripts/app.py` and the same `fill2sr` inference package. It does not call the iOS app service.

## Prepare a review copy

From the repository root:

```bash
python scripts/prepare_space.py --output-dir outputs/huggingface-space
```

The target folder must not already exist. The builder includes the inference source, Gradio UI, 16 example inputs and pinned deployment configuration. It excludes checkpoints, outputs, Git history and local credentials. `BUILD_MANIFEST.json` records the packaged files and SHA-256 hashes. Building the folder does not upload anything or allocate cloud hardware.

## Private setup

1. Create a **private Gradio Space**, such as `yixingfu/Fill2SR-Demo`.
2. Select **ZeroGPU** if available for your account. Do not select paid hardware without first agreeing a budget. Account eligibility and hardware availability are determined by Hugging Face.
3. Obtain access to FLUX.1-Fill-dev, FLUX.1-Redux-dev and the private Fill2SR model repository. Add a read-only `HF_TOKEN` to **Space Settings → Secrets**. Never place the token in code, README, public variables or Git.
4. Upload the *contents* of the prepared folder to the Space root. The Space README supplies `sdk`, `sdk_version`, `python_version` and `app_file`.
5. Wait for the build, startup and checkpoint checksum verification to complete. All downloads are pinned in `models.json`. Only the Diffusers component folders are downloaded for the two base models, avoiding duplicate standalone weights.

The model secret is used by the server to download checkpoints. Visitors do not need to download the weights to run the demo. Space access and GPU usage quotas are separate from model-download gating.

## Runtime

- Default: **2K, 28 steps, seed 2026**, with Nokia first.
- Uploaded files: at most **10 MB**; each image side must be **32–1024 px**, with aspect ratio between **1:4 and 4:1**.
- Scale output limit: **3,686,400 pixels**. Requests exceeding the limit are rejected before acquiring a GPU; choose 2K or a smaller scale. Inputs are not silently resized.
- One generation runs at a time, with a queue of up to eight requests.
- ZeroGPU wraps model inference with `@spaces.GPU(duration=120)`. Weights are placed on CUDA at module startup, following the ZeroGPU loading pattern. Cloud runtime and memory must be measured on the actual Space.
- Model CPU offload is disabled for ZeroGPU. A separately configured dedicated GPU can use `FILL2SR_OFFLOAD=1`; changing this variable does not change the Space's hardware.
- Uploaded/display images use Gradio's temporary cache. Cached files and generated outputs older than an hour are cleaned periodically (every ten minutes). The demo does not create an upload archive or use visitor inputs for training. Generated download links are temporary.
- `share=False`: local tests do not create a public Gradio tunnel.

For local validation with existing checkpoints:

```bash
cd outputs/huggingface-space
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set this to your local model directory; never commit local paths.
export FILL2SR_MODELS_DIR=/path/to/Fill2SR/models
CUDA_VISIBLE_DEVICES=0 python deploy/huggingface/app.py
```

Use a compatible Python 3.12/PyTorch environment for this check. A normal GPU test validates the packaged app and inference path; only a real ZeroGPU run validates its scheduler, CUDA emulation and usage quota behavior.

## Acceptance before public release

- Build from this package and download all three pinned checkpoints using the Space secret.
- Generate Nokia with the defaults and inspect the image and PNG download. GPU/framework differences can affect exact pixels; compare dimensions and visual quality as well as recording hashes.
- Upload a new image, run scale mode, and verify queue behavior and recovery after a rejected request.
- Check input/output limits, cleanup, and absence of credentials in the published files and logs.
- Measure ZeroGPU runtime and memory; adjust the 120-second reservation if required by measured results.
- Review the private Space before making it public or adding its link to the repository and project page.

Model visibility, the Space visibility, GitHub pushes and GitHub Pages deployment are separate release actions.

## References

- [ZeroGPU configuration and compatibility](https://huggingface.co/docs/hub/spaces-zerogpu)
- [Space configuration reference](https://huggingface.co/docs/hub/spaces-config-reference)
- [Space secrets and visibility](https://huggingface.co/docs/hub/spaces-overview)
