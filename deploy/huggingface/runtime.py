"""Checkpoint preparation and resource limits for the hosted research demo."""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

MAX_STEPS = 28
MAX_INPUT_EDGE = 1024
MAX_OUTPUT_PIXELS = 2560 * 1440
MODEL_CONFIG = Path(__file__).with_name('models.json')


def verify_lora(path, expected_sha256):
    digest = hashlib.sha256()
    with Path(path).open('rb') as checkpoint:
        for chunk in iter(lambda: checkpoint.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('Ours-Full checkpoint checksum mismatch; check the model download.')


def prepare_checkpoints(local_dir=None):
    """Download only the required Diffusers files, or validate a local model folder."""
    config = json.loads(MODEL_CONFIG.read_text())
    if local_dir:
        root = Path(local_dir).expanduser().resolve()
        fill, redux, lora = root/'FLUX.1-Fill-dev', root/'FLUX.1-Redux-dev', root/'sr-full.safetensors'
        for folder in (fill, redux):
            if not (folder/'model_index.json').is_file():
                raise FileNotFoundError(f'Missing model_index.json in {folder}')
    else:
        from huggingface_hub import hf_hub_download, snapshot_download
        token = os.environ.get('HF_TOKEN') or None
        fill = Path(snapshot_download(**config['fill'], token=token))
        redux = Path(snapshot_download(**config['redux'], token=token))
        lora = Path(hf_hub_download(**config['lora']['download'], token=token))
    verify_lora(lora, config['lora']['sha256'])
    return fill, redux, lora


def validate_request(image, size, steps):
    """Reject oversized requests before acquiring GPU time; never resize silently."""
    import gradio as gr
    if min(image.size) < 32 or max(image.size) > MAX_INPUT_EDGE:
        raise gr.Error('For the online demo, each input side must be between 32 and 1024 pixels.')
    if max(image.size) / min(image.size) > 4:
        raise gr.Error('For the online demo, use an aspect ratio between 1:4 and 4:1.')
    if not 1 <= steps <= MAX_STEPS:
        raise gr.Error('For the online demo, choose between 1 and 28 sampling steps.')
    if size not in ('2K', '2×', '3×', '4×'):
        raise gr.Error('Choose 2K, 2×, 3×, or 4×.')
    if size != '2K':
        scale = int(size[0])
        if image.width * image.height * scale**2 > MAX_OUTPUT_PIXELS:
            raise gr.Error('This scale exceeds the online demo output limit. Choose 2K or a smaller scale.')


def prune_outputs(directory, *, now=None, max_age=3600):
    """Remove only expired generated files inside this process's output directory."""
    cutoff = (time.time() if now is None else now) - max_age
    for path in Path(directory).glob('fill2sr_*.png'):
        try:
            if not path.is_symlink() and path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except FileNotFoundError:
            pass


def start_output_cleanup(directory, *, interval=600, max_age=3600):
    stop = threading.Event()

    def worker():
        while not stop.wait(interval):
            prune_outputs(directory, max_age=max_age)

    threading.Thread(target=worker, name='fill2sr-output-cleanup', daemon=True).start()
    return stop
