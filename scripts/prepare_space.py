"""Build a small, reviewable Hugging Face Space folder without uploading it."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(destination):
    destination = Path(destination).resolve()
    # Always build into a new folder so existing work cannot be overwritten.
    destination.mkdir(parents=True, exist_ok=False)
    directories = ('src/fill2sr', 'assets/examples/reallq250', 'configs', 'deploy/huggingface')
    for name in directories:
        shutil.copytree(ROOT/name, destination/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (destination/'scripts').mkdir()
    shutil.copy2(ROOT/'scripts/app.py', destination/'scripts/app.py')
    shutil.copy2(ROOT/'LICENSE', destination/'LICENSE')
    shutil.copy2(ROOT/'requirements.txt', destination/'requirements-core.txt')
    for name in ('README.md', 'requirements.txt'):
        shutil.copy2(ROOT/'deploy/huggingface'/name, destination/name)
    (destination/'.gitignore').write_text('__pycache__/\n*.py[cod]\n.env\n.venv/\nmodels/\noutputs/\n*.safetensors\n')
    manifest = {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(destination.rglob('*')) if path.is_file()}
    (destination/'BUILD_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    files = prepare(args.output_dir)
    print(f'Prepared {len(files)} files in {args.output_dir}. No upload performed.')
