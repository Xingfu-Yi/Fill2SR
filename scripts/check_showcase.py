"""Verify the published showcase pairs against their provenance manifest."""
import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'


def main():
    manifest = json.loads((DOCS / 'gallery.json').read_text())
    examples = manifest['examples']
    inputs_dir = ROOT / 'assets/examples/reallq250'
    selection = json.loads((inputs_dir / 'selection.json').read_text())['examples']
    assert len(examples) == 16 and examples[0]['id'] == '113'
    assert [e['id'] for e in examples] == [e['id'] for e in selection]
    assert {(e['paper_figure'], e['paper_row']) for e in examples} == {
        (figure, row) for figure in ('A3', 'A4', 'A5', 'A6') for row in range(1, 5)
    }
    assert len({e['id'] for e in examples}) == len(examples)
    assert manifest['settings'] == {'resolution': '2k', 'steps': 28, 'seed': 2026}
    for item, source in zip(examples, selection):
        for key in ('id', 'title', 'paper_figure', 'paper_row', 'input_sha256'):
            assert item[key] == source[key], (item['id'], key)
        assert item['settings'] == manifest['settings']
        assert item['model_sha256'] == manifest['model_sha256']
        assert len(item['code_revision']) == 40
        bundled = inputs_dir / f"{item['id']}.png"
        assert hashlib.sha256(bundled.read_bytes()).hexdigest() == item['input_sha256']
        for kind in ('input', 'output'):
            path = (DOCS / item[kind]).resolve()
            assert path.is_relative_to(DOCS.resolve()), path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item[f'{kind}_sha256'], path
            with Image.open(path) as image:
                image.load()
                assert list(image.size) == item[f'{kind}_size'], path
                assert image.format == 'PNG', path
        assert item['input_size'] == [256, 256]
        assert item['output_size'] == [1536, 1536]
        with Image.open(DOCS / item['thumbnail']) as image:
            image.load()
            assert image.size == (240, 240)
    assert {p.stem for p in inputs_dir.glob('*.png')} == {e['id'] for e in examples}
    assert {p.name for p in (DOCS / 'assets/gallery').iterdir()} == {
        Path(e[kind]).name for e in examples for kind in ('input', 'output', 'thumbnail')
    }
    default = inputs_dir / '113.png'
    assert hashlib.sha256(default.read_bytes()).hexdigest() == examples[0]['input_sha256']
    figures = json.loads((DOCS / 'paper-figures.json').read_text())['figures']
    source_files = {p.name for p in (ROOT / 'assets').glob('*.png')}
    repository_only = {'real_patch_identity.png', 'boundary_consistency.png', 'ablation_grid.png'}
    assert repository_only <= source_files
    assert {Path(f['source']).name for f in figures} == source_files - repository_only
    html = (DOCS / 'index.html').read_text()
    assert all(Path(name).stem not in html for name in repository_only)
    assert 'figure-actual' not in html
    assert {p.name for p in (DOCS / 'assets/paper').glob('*.webp')} == {
        Path(f['preview']).name for f in figures
    }
    for figure in figures:
        for kind, base in (('source', ROOT), ('preview', DOCS)):
            path = (base / figure[kind]).resolve()
            assert path.is_relative_to(base.resolve()), path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == figure[f'{kind}_sha256'], path
            with Image.open(path) as image:
                image.load()
                assert list(image.size) == figure[f'{kind}_size'], path
        sw, sh = figure['source_size']
        pw, ph = figure['preview_size']
        assert pw <= 2400 and abs(ph - sh * pw / sw) <= 0.5
        assert html.count(f'data-paper-figure="{figure["id"]}"') == 1
        assert figure['preview'] in html
        if figure.get('original_pdf'):
            pdf = ROOT / figure['original_pdf']
            assert hashlib.sha256(pdf.read_bytes()).hexdigest() == figure['original_pdf_sha256'], pdf
    assert {f.get('arxiv_appendix_figure') for f in figures if f.get('arxiv_appendix_figure')} == {f'A{i}' for i in range(1, 9)}
    assert 'role="tab"' not in html and 'role="tabpanel"' not in html
    print(f'Verified {len(figures)} paper figures: source/preview hashes, full-frame dimensions and HTML coverage.')
    print('Verified 16 paper pairs: hashes, dimensions, per-image settings, figure references, Gradio inputs and Nokia default.')


if __name__ == '__main__':
    main()
