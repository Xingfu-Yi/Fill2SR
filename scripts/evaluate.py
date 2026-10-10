"""Generate Ours-Full predictions for an image directory, preserving relative paths."""

import argparse
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from fill2sr import Fill2SRPipeline
from fill2sr.paths import checkpoint_paths

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    size = parser.add_mutually_exclusive_group()
    size.add_argument('--scale', type=float, default=2.0)
    size.add_argument('--resolution', choices=['2k', '4k', '5k', '8k', '12k'])
    parser.add_argument('--steps', type=int, default=28)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--dry-run', action='store_true', help='List inputs and outputs without loading models')
    parser.add_argument('--fill-dir', type=Path)
    parser.add_argument('--redux-dir', type=Path)
    parser.add_argument('--lora', type=Path)
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f'--input-dir does not exist: {args.input_dir}')
    if args.scale <= 0 or args.steps < 1 or (args.max_images is not None and args.max_images < 1):
        parser.error('--scale, --steps and --max-images must be positive')
    inputs = sorted(p for p in args.input_dir.rglob('*')
                    if p.is_file() and p.suffix.lower() in EXTENSIONS
                    and not any(part.startswith('.') for part in p.relative_to(args.input_dir).parts))
    if args.max_images is not None:
        inputs = inputs[:args.max_images]
    if not inputs:
        parser.error('No input images found')
    for source in inputs:
        try:
            with Image.open(source) as image:
                image.load()
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            parser.error(f'Cannot read input image {source}: {exc}')
    jobs = [(p, args.output_dir / p.relative_to(args.input_dir).with_suffix('.png')) for p in inputs]
    sources_by_target = {}
    for source, target in jobs:
        sources_by_target.setdefault(target.resolve(), []).append(source)
    conflicts = {target: sources for target, sources in sources_by_target.items() if len(sources) > 1}
    if conflicts:
        details = '\n'.join(f'{target} <- {", ".join(map(str, sources))}'
                            for target, sources in conflicts.items())
        parser.error(f'Input images map to the same output path; rename the conflicting inputs:\n{details}')
    if args.dry_run:
        for source, target in jobs:
            print(f'{source} -> {target}')
        print(f'{len(jobs)} image(s)')
        return
    fill, redux, lora = checkpoint_paths(args.fill_dir, args.redux_dir, args.lora)
    model = Fill2SRPipeline(fill, redux, lora, off_load=args.offload, device=args.device)
    for index, (source, target) in enumerate(jobs, 1):
        with Image.open(source) as image:
            result = model(image.convert('RGB'),
                           upscale=args.scale if args.resolution is None else None,
                           resolution=args.resolution, sample_steps=args.steps, seed=args.seed)
        target.parent.mkdir(parents=True, exist_ok=True)
        result.save(target)
        print(f'[{index}/{len(jobs)}] {target}')


if __name__ == '__main__':
    main()
