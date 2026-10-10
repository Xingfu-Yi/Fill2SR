"""Hugging Face Spaces entry point for the ECCV 2026 Ours-Full model."""

# Import before torch/diffusers so ZeroGPU can initialize CUDA emulation.
import spaces

import atexit
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))

TEMP = tempfile.TemporaryDirectory(prefix='fill2sr-space-')
OUTPUTS = Path(TEMP.name)/'outputs'
OUTPUTS.mkdir()
os.environ.setdefault('GRADIO_TEMP_DIR', str(Path(TEMP.name)/'cache'))

from deploy.huggingface.runtime import MAX_STEPS, prepare_checkpoints, start_output_cleanup, validate_request
from fill2sr import Fill2SRPipeline
from scripts.app import EXAMPLES_DIR, build_demo

IS_ZERO_GPU = os.environ.get('SPACES_ZERO_GPU', '').lower() in {'1', 'true', 'yes', 'on'}
offload = os.environ.get('FILL2SR_OFFLOAD', '0') == '1'
if IS_ZERO_GPU and offload:
    raise ValueError('ZeroGPU uses resident CUDA models; unset FILL2SR_OFFLOAD for this Space.')

fill, redux, lora = prepare_checkpoints(os.environ.get('FILL2SR_MODELS_DIR'))
model = Fill2SRPipeline(fill, redux, lora, device='cuda:0', off_load=offload)


@spaces.GPU(duration=120)
def restore(image, **kwargs):
    return model(image, **kwargs)


cleanup_stop = start_output_cleanup(OUTPUTS)
atexit.register(cleanup_stop.set)
demo = build_demo(
    restore, output_dir=OUTPUTS, request_validator=validate_request,
    max_steps=MAX_STEPS, delete_cache=(600, 3600),
    usage_note='Online demo · Inputs up to 1024 × 1024 px · Up to 28 steps · '
               'Large scale outputs are limited; choose 2K for larger inputs.\n\n'
               'Uploaded images and results are temporarily cached. Files older than one hour '
               'are periodically removed; download your result after generation.',
)

if __name__ == '__main__':
    demo.launch(
        server_name='0.0.0.0' if os.environ.get('SPACE_ID') else '127.0.0.1',
        server_port=int(os.environ.get('PORT', '7860')),
        share=False, show_error=False, max_file_size='10mb',
        allowed_paths=[str(EXAMPLES_DIR), str(OUTPUTS)],
    )
