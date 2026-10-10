"""Make uncropped web previews of the original paper figures; keep PNGs intact."""
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    'teaser', 'qingming_12k', 'real_world_comparison', 'synthetic_comparison',
    'real_world_examples_1', 'real_world_examples_2', 'real_world_examples_3',
    'real_world_examples_4', 'real_world_examples_5', 'rcdt_pipeline',
)
PDF_SOURCES = {
    'teaser': ('teaser_ultimate', None),
    'qingming_12k': ('supp_12k_2row_grid', 'A1'),
    'rcdt_pipeline': ('RCDT_optimized', 'A2'),
    'real_world_examples_4': ('full_vs_grid_page1', 'A3'),
    'real_world_examples_1': ('full_vs_grid_page2', 'A4'),
    'real_world_examples_2': ('full_vs_grid_page4', 'A5'),
    'real_world_examples_3': ('full_vs_grid_page5', 'A6'),
    'real_world_examples_5': ('full_vs_grid_page6', 'A7'),
    'synthetic_comparison': ('synthetic_comparison', 'A8'),
    'real_world_comparison': ('real_world_comparison', None),
}


def main():
    folder = ROOT / 'docs/assets/paper'
    folder.mkdir(parents=True, exist_ok=True)
    figures = []
    for name in NAMES:
        source = ROOT / 'assets' / f'{name}.png'
        target = folder / f'{name}.webp'
        with Image.open(source) as image:
            original_size = list(image.size)
            preview = image.convert('RGB')
            width = min(image.width, 2400)
            preview = preview.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
            preview.save(target, 'WEBP', quality=94, method=6)
        figures.append({
            'id': name, 'source': source.relative_to(ROOT).as_posix(),
            'source_size': original_size,
            'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'preview': target.relative_to(ROOT / 'docs').as_posix(),
            'preview_size': list(preview.size),
            'preview_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        })
        if name in PDF_SOURCES:
            pdf_name, appendix = PDF_SOURCES[name]
            pdf = ROOT / 'results' / f'{pdf_name}.pdf'
            figures[-1].update({
                'original_pdf': pdf.relative_to(ROOT).as_posix(),
                'original_pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
                'arxiv_appendix_figure': appendix,
            })
    (ROOT / 'docs/paper-figures.json').write_text(json.dumps({
        'description': 'Uncropped display derivatives of the original paper figures. PNG originals remain in assets/.',
        'conversion': 'RGB, full-frame Lanczos resize to at most 2400 px wide, WebP quality 94.',
        'figures': figures,
    }, indent=2) + '\n')
    print(f'Prepared {len(figures)} complete paper figures ({sum(p.stat().st_size for p in folder.glob("*.webp")) / 1e6:.1f} MB).')


if __name__ == '__main__':
    main()
