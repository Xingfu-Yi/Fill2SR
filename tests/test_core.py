"""Inference geometry, device placement and recovery without model weights."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

from fill2sr.core import Fill2SRPipeline, inference_device, table
from fill2sr.pipelines.pipeline_flux_fill2sr import LocalAttentionPlus
from fill2sr.tile_geometry import build_visit_count_map, compute_tile_boxes


class GeometryImage:
    """Exercise large output dimensions without allocating large image buffers."""
    def __init__(self, size):
        self.size = size
        self.width, self.height = size

    def resize(self, size, resample=None):
        return GeometryImage(size)

    def crop(self, box):
        return GeometryImage((box[2] - box[0], box[3] - box[1]))


class GeometryPipeline:
    vae_scale_factor = 8

    def __init__(self):
        self.vae = SimpleNamespace(use_tiling=False)
        self.calls = []
        self.fail = False

    def enable_vae_tiling(self):
        self.vae.use_tiling = True

    def disable_vae_tiling(self):
        self.vae.use_tiling = False

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError('simulated inference failure')
        return SimpleNamespace(images=[GeometryImage((kwargs['width'], kwargs['height']))])


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.model = Fill2SRPipeline.__new__(Fill2SRPipeline)
        self.model.pipe = GeometryPipeline()
        self.model.redux = None
        self.model.system_prompt_sr = 'test'

    def test_high_resolution_keeps_canvas_and_covers_all_latents(self):
        for shape in ((256, 256), (320, 180), (180, 320)):
            for preset in ('4k', '5k', '8k', '12k'):
                with self.subTest(shape=shape, preset=preset):
                    output = self.model(GeometryImage(shape), resolution=preset)
                    ratio = math.sqrt(table[preset] / math.prod(shape))
                    expected = tuple(math.ceil(int(side * ratio) / 16) * 16 for side in shape)
                    self.assertEqual(output.size, expected)
                    call = self.model.pipe.calls[-1]
                    self.assertEqual(call['image'].size, expected)
                    w, h = expected
                    kw, kh = call['kernel_width'], call['kernel_height']
                    self.assertGreater(w * h, kw * kh)
                    pixel_boxes = compute_tile_boxes(h, w, kh, kw, 0.75)
                    latent_boxes = compute_tile_boxes(h // 8, w // 8, kh // 8, kw // 8, 0.75)
                    self.assertEqual(pixel_boxes, [tuple(v * 8 for v in box) for box in latent_boxes])
                    coverage = build_visit_count_map(h // 8, w // 8, kh // 8, kw // 8, 0.75)
                    self.assertGreater(coverage.min(), 0)

    def test_2k_nokia_geometry_is_unchanged(self):
        output = self.model(Image.new('RGB', (256, 256)), resolution='2k')
        self.assertEqual(output.size, (1536, 1536))
        self.assertEqual(self.model.pipe.calls[-1]['image'].size, (1536, 1536))

    def test_small_scale_remains_exact(self):
        for scale in (2, 3, 4):
            with self.subTest(scale=scale):
                output = self.model(Image.new('RGB', (257, 193)), upscale=scale)
                self.assertEqual(output.size, (257 * scale, 193 * scale))

    def test_large_scale_uses_full_canvas_before_final_alignment(self):
        source = GeometryImage((1001, 701))
        output = self.model(source, upscale=4)
        self.assertEqual(output.size, (4004, 2804))
        call = self.model.pipe.calls[-1]
        self.assertEqual((call['width'], call['height']), (4016, 2816))

    def test_tiling_state_restored_after_failure_and_success(self):
        for original in (False, True):
            for fail in (False, True):
                with self.subTest(original=original, fail=fail):
                    self.model.pipe.vae.use_tiling = original
                    self.model.pipe.fail = fail
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'simulated'):
                            self.model(GeometryImage((256, 256)), resolution='4k')
                    else:
                        self.model(GeometryImage((256, 256)), resolution='4k')
                    self.assertEqual(self.model.pipe.vae.use_tiling, original)
        self.model.pipe.fail = False
        self.model.pipe.vae.use_tiling = False
        self.assertEqual(self.model(Image.new('RGB', (256, 256)), resolution='2k').size, (1536, 1536))

    def test_cpu_tiles_reconstruct_without_default_cuda_allocation(self):
        tiles = LocalAttentionPlus(kernel_size=(32, 48), overlap=0.75, channel_dim=3)
        source = torch.rand(1, 3, 80, 112)
        patches = tiles.grids(source)
        self.assertEqual(tiles.tile_weights.device, source.device)
        restored = tiles.grids_inverse(patches)
        torch.testing.assert_close(restored, source, rtol=1e-5, atol=1e-6)


class DeviceTest(unittest.TestCase):
    def test_offload_routes_both_models_to_requested_gpu(self):
        fill, redux = Mock(), Mock()
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=4), \
                patch('fill2sr.core.FluxFill2SRPipeline.from_pretrained', return_value=fill), \
                patch('fill2sr.core.FluxPriorReduxPipeline.from_pretrained', return_value=redux):
            model = Fill2SRPipeline('fill', 'redux', 'lora', device='cuda:2', off_load=True)
        self.assertEqual(model.device, 'cuda:2')
        fill.enable_model_cpu_offload.assert_called_once_with(device='cuda:2')
        redux.enable_model_cpu_offload.assert_called_once_with(device='cuda:2')

    def test_invalid_device_fails_before_checkpoints_are_loaded(self):
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2), \
                patch('fill2sr.core.FluxFill2SRPipeline.from_pretrained') as factory:
            for device in ('cuda:7', 'not-a-device', 'cpu'):
                with self.subTest(device=device), self.assertRaises(ValueError):
                    Fill2SRPipeline('fill', 'redux', 'lora', device=device, off_load=True)
            factory.assert_not_called()

    def test_default_cuda_resolves_to_zero(self):
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=4):
            self.assertEqual(inference_device('cuda', True), 'cuda:0')


if __name__ == '__main__':
    unittest.main()
