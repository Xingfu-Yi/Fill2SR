"""Batch CLI behavior without downloading or loading model weights."""
import contextlib
import io
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate.py'


class BatchCLITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        self.output = self.root / 'output'
        Image.new('RGB', (16, 16)).save(self.inputs / 'visible.png')
        self.pipeline = Mock(return_value=Image.new('RGB', (32, 32)))
        self.factory = Mock(return_value=self.pipeline)

    def run_cli(self, *args):
        package = types.ModuleType('fill2sr')
        package.Fill2SRPipeline = self.factory
        paths = types.ModuleType('fill2sr.paths')
        paths.checkpoint_paths = Mock(return_value=('fill', 'redux', 'lora'))
        argv = [str(SCRIPT), '--input-dir', str(self.inputs), '--output-dir', str(self.output), *args]
        out, err = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {'fill2sr': package, 'fill2sr.paths': paths}), patch.object(sys, 'argv', argv), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                runpy.run_path(str(SCRIPT), run_name='__main__')
            except SystemExit as exc:
                return exc.code, out.getvalue(), err.getvalue()
        return 0, out.getvalue(), err.getvalue()

    def test_default_remains_two_times(self):
        self.assertEqual(self.run_cli()[0], 0)
        args = self.pipeline.call_args.kwargs
        self.assertEqual(args['upscale'], 2)
        self.assertIsNone(args['resolution'])
        self.assertTrue((self.output / 'visible.png').is_file())

    def test_resolution_is_forwarded_without_scale(self):
        self.assertEqual(self.run_cli('--resolution', '2k', '--steps', '28', '--seed', '2026')[0], 0)
        args = self.pipeline.call_args.kwargs
        self.assertIsNone(args['upscale'])
        self.assertEqual((args['resolution'], args['sample_steps'], args['seed']), ('2k', 28, 2026))

    def test_size_modes_are_mutually_exclusive(self):
        code, _, error = self.run_cli('--scale', '2', '--resolution', '2k')
        self.assertEqual(code, 2)
        self.assertIn('not allowed with argument', error)
        self.factory.assert_not_called()

    def test_hidden_sidecars_and_directories_are_skipped(self):
        (self.inputs / '._195.png').write_text('not an image')
        (self.inputs / '.hidden').mkdir()
        (self.inputs / '.hidden' / 'broken.png').write_text('not an image')
        (self.inputs / 'nested').mkdir()
        Image.new('RGB', (16, 16)).save(self.inputs / 'nested' / 'photo.jpg')
        code, output, _ = self.run_cli('--resolution', '2k', '--dry-run')
        self.assertEqual(code, 0)
        self.assertIn('2 image(s)', output)
        self.assertIn(str(self.output / 'nested' / 'photo.png'), output)
        self.assertNotIn('._195', output)
        self.factory.assert_not_called()

    def test_unreadable_file_is_named_before_model_load(self):
        (self.inputs / 'broken.png').write_text('not an image')
        code, _, error = self.run_cli()
        self.assertEqual(code, 2)
        self.assertIn('Cannot read input image', error)
        self.assertIn('broken.png', error)
        self.factory.assert_not_called()

    def test_colliding_extensions_fail_without_writing_outputs(self):
        Image.new('RGB', (16, 16)).save(self.inputs / 'visible.jpg')
        self.output.mkdir()
        existing = self.output / 'visible.png'
        existing.write_bytes(b'previous output')
        for flags in ((), ('--dry-run',)):
            with self.subTest(flags=flags):
                code, _, error = self.run_cli(*flags)
                self.assertEqual(code, 2)
                self.assertIn('same output path', error)
                self.assertIn('visible.jpg', error)
                self.assertIn('visible.png', error)
                self.assertEqual(existing.read_bytes(), b'previous output')
                self.factory.assert_not_called()

    def test_same_stem_in_separate_directories_is_allowed(self):
        (self.inputs / 'nested').mkdir()
        Image.new('RGB', (16, 16)).save(self.inputs / 'nested' / 'visible.jpg')
        self.assertEqual(self.run_cli()[0], 0)
        self.assertEqual(self.pipeline.call_count, 2)
        self.assertTrue((self.output / 'visible.png').exists())
        self.assertTrue((self.output / 'nested' / 'visible.png').exists())


if __name__ == '__main__':
    unittest.main()
