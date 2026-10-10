import random
import math
from PIL import Image
import logging
import time
import gc
from typing import List, Optional
import numpy as np
import torch
from diffusers import FluxPriorReduxPipeline
from .pipelines import FluxFill2SRPipeline
from .utils import get_image_resolutions

logger = logging.getLogger(__name__)

ASPECT_RATIOS = [16 / 9, 4 / 3, 1 / 1, 3 / 4, 9 / 16]
SIZE_LIST = [1536, 1920, 2048, 2560]
MIN_RES = 1536 * 864
MAX_RES = 2560 * 1440

table = {
    "2k": 2560 * 1440,
    "4k": 3840 * 2160,
    "5k": 5120 * 2880,
    "8k": 7680 * 4320,
    "12k": 12288 * 6480,
}

def setup_logging(log_level):
    """Configure only Fill2SR logs, preserving the host application's logging."""
    package_logger = logging.getLogger("fill2sr")
    package_logger.setLevel(log_level)
    package_logger.propagate = False
    if not package_logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        package_logger.addHandler(stream_handler)


def inference_device(device, off_load):
    """Validate the requested device before loading any checkpoints."""
    try:
        resolved = torch.device(device)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid inference device: {device}") from exc
    if off_load and resolved.type != "cuda":
        raise ValueError("CPU offload requires a CUDA device, such as cuda:0")
    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device cuda:{index} is unavailable; check --device and CUDA_VISIBLE_DEVICES")
        resolved = torch.device("cuda", index)
    elif resolved.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS device is unavailable")
    elif resolved.type != "cpu":
        raise ValueError(f"Unsupported inference device: {resolved}")
    return str(resolved)


def resize_and_centercrop(image: Image.Image, size: tuple) -> Image.Image:
    w, h = image.size
    W, H = size
    ratio = max(W / w, H / h)
    w, h = int(w * ratio), int(h * ratio)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    left = (w - W) // 2
    top = (h - H) // 2
    right = left + W
    bottom = top + H
    image = image.crop((left, top, right, bottom))
    return image


def resize_and_fill(image: Image.Image, size: tuple) -> Image.Image:
    w, h = image.size
    W, H = size
    ratio = min(W / w, H / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    image = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    left = (W - new_w) // 2
    top = (H - new_h) // 2
    right = left + new_w
    bottom = top + new_h
    fill_image = Image.new("RGB", size, "white")
    crop_rect = (left, top, right, bottom)
    fill_image.paste(image, crop_rect)
    return fill_image, crop_rect


class Fill2SRPipeline:
    def __init__(
        self,
        flux_fill_dir,
        flux_redux_dir,
        lora_path,
        off_load=True,
        torch_dtype=torch.bfloat16,
        device="cuda",
        debug=False,
    ):
        log_level = logging.DEBUG if debug else logging.WARNING
        setup_logging(log_level=log_level)
        self.debug = debug
        self.dtype = torch_dtype
        self.device = inference_device(device, off_load)
        self.lora_path = str(lora_path)

        self.pipe = FluxFill2SRPipeline.from_pretrained(flux_fill_dir, torch_dtype=self.dtype)
        self.pipe.load_lora_weights(self.lora_path)

        self.redux = FluxPriorReduxPipeline.from_pretrained(
            flux_redux_dir,
            tokenizer=self.pipe.tokenizer,
            tokenizer_2=self.pipe.tokenizer_2,
            text_encoder=self.pipe.text_encoder,
            text_encoder_2=self.pipe.text_encoder_2,
            torch_dtype=self.dtype,
        )

        if off_load:
            logger.debug("Enable model CPU offload")
            self.pipe.enable_model_cpu_offload(device=self.device)
            self.redux.enable_model_cpu_offload(device=self.device)
        else:
            self.pipe.to(self.device)
            self.redux.to(self.device)

        self.system_prompt_sr = "Produce clean, sharp, noise-free images at professional DSLR quality. Ensure images are high-res, blur-free, grain-free, artifact-free, distortion-free, with balanced lighting, natural tones, smooth details, and consistent focus. Target 4K/8K clarity. Avoid fuzzy, pixelated, or low-quality outputs."

    @staticmethod
    def fit_image_kernel(image: Image.Image, width: int, height: int):
        closet_ratio, resolutions = get_image_resolutions(width=width, height=height)
        if width * height <= MAX_RES:
            for w, h in resolutions:
                if w == width and h == height:
                    kernel_width = width
                    kernel_height = height
                    grid1 = True
                    return width, height, kernel_width, kernel_height
        resolutions = sorted(resolutions, key=lambda x: min(abs(x[0] - width), abs(x[1] - height)))
        resolutions = [res for res in resolutions if res[0] * res[1] <= MAX_RES]
        kernel_width, kernel_height = resolutions[0]
        ratio = max(kernel_width / image.width, kernel_height / image.height)
        width, height = int(image.width * ratio), int(image.height * ratio)
        width, height = width // 32 * 32, height // 32 * 32
        return width, height, kernel_width, kernel_height

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        width: Optional[int] = None,
        height: Optional[int] = None,
        kernel_width: Optional[int] = None,
        kernel_height: Optional[int] = None,
        upscale: Optional[float] = None,
        resolution: Optional[str] = None,
        overlap: float = 0.75,
        sample_steps: int = 28,
        seed: int = -1,
    ) -> Image.Image:
        """Run Fill2SR super-resolution with automatic sizing and optional tiling.

        The method resolves output dimensions from (width, height) or (upscale | resolution),
        performs optional latent tiling when resolution exceeds `MAX_RES`, and calls the
        underlying diffusion pipeline.

        Args:
            image: Input PIL image.
            width: Optional explicit output width (must be paired with `height`).
            height: Optional explicit output height (must be paired with `width`).
            kernel_width: Optional explicit tile width. If omitted, the pipeline resolves it automatically.
            kernel_height: Optional explicit tile height. If omitted, the pipeline resolves it automatically.
            upscale: Optional scale factor (e.g., 2.0 => 2×). Ignored if width/height given.
            resolution: Optional resolution preset key. Looks up total target pixels in
                the resolution table (e.g., '4k': 3840*2160). Ignored if
                width/height or upscale given.
            overlap: Tile stepping ratio used by the latent tiling path.
            sample_steps: Number of inference steps for the diffusion sampler (> 0).
            seed: Random seed. If < 0, a random seed will be generated.

        Returns:
            The generated/restored PIL image.

        Raises:
            ValueError: On invalid arguments or missing/unsupported resolution preset.
        """
        gc.collect()
        torch.cuda.empty_cache()
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        logger.debug(f"Seed = {seed}")
        generator = torch.Generator("cpu").manual_seed(seed)

        if image is None or getattr(image, "width", 0) <= 0 or getattr(image, "height", 0) <= 0:
            raise ValueError("Input image is missing or has invalid dimensions")
        raw_image_size = image.size
        final_size = None
        system_prompt = self.system_prompt_sr
        if sample_steps < 1:
            raise ValueError("sample_steps must be positive")
        if not 0 < overlap <= 1:
            raise ValueError("overlap (tile stepping ratio) must be between 0 and 1")

        if (width is None) ^ (height is None):
            raise ValueError("width and height must be provided together")
        if (kernel_width is None) ^ (kernel_height is None):
            raise ValueError("kernel_width and kernel_height must be provided together")

        if width is None and height is None:
            if upscale is not None:
                # Validate upscale.
                upscale_f = float(upscale)
                if not math.isfinite(upscale_f) or upscale_f <= 0:
                    raise ValueError("upscale must be > 0")

                # Compute the user-expected final size (force at the end).
                final_size = (
                    int(round(raw_image_size[0] * upscale_f)),
                    int(round(raw_image_size[1] * upscale_f)),
                )

                # Use the same target size as the working size (may be adjusted later by MIN_RES/MAX_RES logic).
                width, height = final_size
            elif resolution is not None:
                if not resolution in table.keys():
                    raise ValueError(f"resolution not supported, please input one of: {table.keys()}")
                pixels = table.get(resolution)
                ratio = np.sqrt(pixels / (image.width * image.height))
                width, height = int(image.width * ratio), int(image.height * ratio)
            else:
                raise ValueError("one of upscale or resolution must be provided, or you can directly input output width and height")
        else:
            # Explicit width/height provided -> do NOT force upscale final resize.
            final_size = None
            width, height = int(width), int(height)

        if width <= 0 or height <= 0:
            raise ValueError("Output width and height must be positive")

        explicit_kernel = kernel_width is not None and kernel_height is not None
        if explicit_kernel:
            kernel_width = int(kernel_width)
            kernel_height = int(kernel_height)
            if kernel_width <= 0 or kernel_height <= 0:
                raise ValueError("kernel_width and kernel_height must be positive")

        # Keep the full target canvas when tiling. Only round its dimensions to
        # the VAE/latent-packing multiple; tile size must never rescale the image.
        alignment = self.pipe.vae_scale_factor * 2
        tiled_target = width * height > MAX_RES or (
            explicit_kernel and (width > kernel_width or height > kernel_height)
        )
        if tiled_target:
            width = max(alignment, math.ceil(width / alignment) * alignment)
            height = max(alignment, math.ceil(height / alignment) * alignment)
        if explicit_kernel and (kernel_width % alignment or kernel_height % alignment):
            raise ValueError(f"Tile dimensions must be divisible by {alignment}")

        if width * height < MIN_RES:
            ratio = np.sqrt(MIN_RES / (width * height))
            width, height = int(width * ratio), int(height * ratio)

        if explicit_kernel:
            latent_tiling = width > kernel_width or height > kernel_height or width * height > MAX_RES
            logger.debug(
                "raw image: %s, explicit target: (%dx%d), kernel: (%dx%d)",
                image.size,
                width,
                height,
                kernel_width,
                kernel_height,
            )
            if latent_tiling:
                image = image.resize((width, height), resample=Image.Resampling.BICUBIC)
                logger.debug("using explicit tiled geometry, resized image: %s", image.size)
            else:
                image, crop_rect = resize_and_fill(image, size=(width, height))
                logger.debug("using explicit full geometry, fill image to target size: %s", image.size)
        else:
            latent_tiling = width * height > MAX_RES
            _, _, kernel_width, kernel_height = self.fit_image_kernel(image=image, width=width, height=height)
            logger.debug(f"raw image: {image.size}, kernel: ({kernel_width}x{kernel_height})")
            if latent_tiling:
                if width > height > 2048:
                    kernel_height = 2048
                    kernel_width = 1536
                elif height > width > 2048:
                    kernel_height = 1536
                    kernel_width = 2048
                image = image.resize((width, height), resample=Image.Resampling.BICUBIC)
                logger.debug(f"adjusted kernel: ({kernel_width}x{kernel_height}) resized image: {image.size}")
            else:
                image, crop_rect = resize_and_fill(image, size=(kernel_width, kernel_height))
                width = kernel_width
                height = kernel_height
                logger.debug(f"keep image aspect ratio, fill image to kernel size: {image.size}")

        previous_tiling = self.pipe.vae.use_tiling
        try:
            if latent_tiling:
                self.pipe.enable_vae_tiling()
            output_image = self.pipe(
                redux=self.redux,
                prompt=system_prompt,
                image=image,
                height=height,
                width=width,
                kernel_width=kernel_width,
                kernel_height=kernel_height,
                overlap=overlap,
                guidance_scale=50,
                num_inference_steps=sample_steps,
                generator=generator,
            ).images[0]
        finally:
            if latent_tiling and not previous_tiling:
                self.pipe.disable_vae_tiling()
        if not latent_tiling:
            output_image = output_image.crop(crop_rect)

        if final_size is not None:
            # Force exact final dimensions for the upscale path only.
            if output_image.size != final_size:
                output_image = output_image.resize(final_size, resample=Image.Resampling.LANCZOS)
            logger.debug(f"raw_size: {raw_image_size}, final_size: {final_size}")

        return output_image
