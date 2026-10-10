"""Run Fill2SR Ours-Full on one image."""

import argparse
from pathlib import Path
from PIL import Image
from fill2sr import Fill2SRPipeline
from fill2sr.paths import checkpoint_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    size = parser.add_mutually_exclusive_group()
    size.add_argument('--scale', type=float, default=2.0)
    size.add_argument('--resolution', choices=['2k', '4k', '5k', '8k', '12k'])
    parser.add_argument('--steps', type=int, default=28)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--fill-dir', type=Path)
    parser.add_argument('--redux-dir', type=Path)
    parser.add_argument('--lora', type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.scale is not None and args.scale <= 0:
        parser.error('--scale must be positive')
    fill, redux, lora = checkpoint_paths(args.fill_dir, args.redux_dir, args.lora)
    model = Fill2SRPipeline(fill, redux, lora, off_load=args.offload, device=args.device)
    with Image.open(args.input) as source:
        result = model(source.convert('RGB'), upscale=args.scale if args.resolution is None else None,
                       resolution=args.resolution, sample_steps=args.steps, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f'Saved {args.output.resolve()} ({result.width}x{result.height})')


if __name__ == '__main__':
    main()
