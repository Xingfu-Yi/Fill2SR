"""Hosted demo limits, checkpoint integrity and packaging without loading weights."""

import hashlib
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import gradio as gr
from PIL import Image

from deploy.huggingface.runtime import prepare_checkpoints, prune_outputs, validate_request, verify_lora
from scripts.app import build_demo, make_generate
from scripts.prepare_space import prepare


class HostingTest(unittest.TestCase):
    def test_nokia_defaults_and_scales_are_accepted(self):
        image = Image.new('RGB', (256, 256))
        for size in ['2K', '2×', '3×', '4×']:
            validate_request(image, size, 28)

    def test_rejected_requests_do_not_acquire_gpu_or_write_outputs(self):
        for shape, size, steps in [((1025, 512), '2K', 28), ((31, 256), '2K', 28),
                                   ((1024, 128), '2K', 28), ((1024, 1024), '4×', 28),
                                   ((256, 256), '2K', 29)]:
            with self.subTest(shape=shape, size=size, steps=steps), tempfile.TemporaryDirectory() as tmp:
                model = Mock()
                generate = make_generate(model, Path(tmp), validate_request)
                with self.assertRaises(gr.Error):
                    list(generate(Image.new('RGB', shape), size, steps, 2026))
                model.assert_not_called()
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_hosted_generation_preserves_parameters_and_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Image.new('RGB', (256, 256), '#aabbcc')
            restored = Image.new('RGB', (1536, 1536), '#112233')
            model = Mock(return_value=restored)
            output = list(make_generate(model, Path(tmp), validate_request)(image, '2K', 28, 2026))[-1]
            args, kwargs = model.call_args
            self.assertEqual(args[0].size, image.size)
            self.assertEqual(kwargs, {'resolution': '2k', 'sample_steps': 28, 'seed': 2026})
            with Image.open(output[2]['value']) as saved:
                self.assertEqual(saved.tobytes(), restored.tobytes())
            self.assertIn('Seed 2026', output[3])

    def test_gpu_quota_error_is_preserved(self):
        model = Mock(side_effect=gr.Error('GPU quota exhausted'))
        with tempfile.TemporaryDirectory() as tmp:
            generate = make_generate(model, Path(tmp), validate_request)
            with self.assertRaisesRegex(gr.Error, 'GPU quota exhausted'):
                list(generate(Image.new('RGB', (256, 256)), '2K', 28, 2026))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_lora_integrity_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sr-full.safetensors'
            path.write_bytes(b'checkpoint')
            verify_lora(path, hashlib.sha256(b'checkpoint').hexdigest())
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_lora(path, '0'*64)

    def test_downloads_use_pinned_revisions_and_read_token(self):
        config = json.loads((Path(__file__).resolve().parents[1]/'deploy/huggingface/models.json').read_text())
        with patch.dict(os.environ, {'HF_TOKEN': 'test-only-token'}), \
             patch('huggingface_hub.snapshot_download', side_effect=['/cache/fill', '/cache/redux']) as snapshot, \
             patch('huggingface_hub.hf_hub_download', return_value='/cache/sr-full.safetensors') as download, \
             patch('deploy.huggingface.runtime.verify_lora') as verify:
            self.assertEqual(prepare_checkpoints(), (Path('/cache/fill'), Path('/cache/redux'), Path('/cache/sr-full.safetensors')))
            for call, key in zip(snapshot.call_args_list, ['fill', 'redux']):
                self.assertEqual(call.kwargs, {**config[key], 'token': 'test-only-token'})
                self.assertRegex(call.kwargs['revision'], r'^[a-f0-9]{40}$')
            download.assert_called_once_with(**config['lora']['download'], token='test-only-token')
            verify.assert_called_once_with(Path('/cache/sr-full.safetensors'), config['lora']['sha256'])

    def test_cleanup_preserves_recent_files_and_symlink_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root/'fill2sr_old.png'
            fresh = root/'fill2sr_new.png'
            other = root/'example.png'
            for path in (old, fresh, other):
                path.write_bytes(b'image')
            (root/'fill2sr_link.png').symlink_to(other)
            now = time.time()
            os.utime(old, (now-4000, now-4000))
            prune_outputs(root, now=now)
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())
            self.assertTrue(other.exists())
            self.assertTrue((root/'fill2sr_link.png').is_symlink())

    def test_package_is_small_self_contained_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)/'space'
            manifest = prepare(destination)
            self.assertTrue((destination/'deploy/huggingface/app.py').is_file())
            self.assertTrue((destination/'src/fill2sr/core.py').is_file())
            self.assertEqual(len(list((destination/'assets/examples/reallq250').glob('*.png'))), 16)
            self.assertFalse(any(name.endswith('.safetensors') or '.git/' in name for name in manifest))
            for name, expected in manifest.items():
                self.assertEqual(hashlib.sha256((destination/name).read_bytes()).hexdigest(), expected)
            with self.assertRaises(FileExistsError):
                prepare(destination)

    def test_hosted_controls_keep_defaults_and_queue(self):
        demo = build_demo(Mock(), request_validator=validate_request, max_steps=28, delete_cache=(600, 3600))
        config = demo.get_config_file()
        steps = next(c for c in config['components'] if c['type']=='slider')
        self.assertEqual((steps['props']['value'], steps['props']['maximum']), (28, 28))
        dataset = next(c for c in config['components'] if c['type']=='dataset')
        self.assertEqual(len(dataset['props']['samples']), 16)
        self.assertIn('113.png', str(dataset['props']['samples'][0]))
        self.assertTrue(any(d['api_name']=='generate' for d in config['dependencies']))
        self.assertEqual(demo._queue.max_size, 8)
        demo.close()


if __name__ == '__main__':
    unittest.main()
