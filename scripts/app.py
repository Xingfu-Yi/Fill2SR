"""Interactive Gradio demo for Fill2SR Ours-Full."""

import argparse
import json
import logging
import secrets
import time
from pathlib import Path
from uuid import uuid4

import gradio as gr
from PIL import Image, ImageOps

from fill2sr.paths import PROJECT_ROOT, checkpoint_paths

EXAMPLES_DIR = PROJECT_ROOT / 'assets' / 'examples' / 'reallq250'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs' / 'gradio'
EXAMPLES = [(item['id'] + '.png', item['title']) for item in
            json.loads((EXAMPLES_DIR / 'selection.json').read_text())['examples']]
READY = 'Choose an example or upload an image, then select **Generate**.'

CSS = """
.gradio-container { width: 100% !important; max-width: 1280px !important;
    margin: 0 auto !important; padding: 28px clamp(16px, 2vw, 24px) 32px !important; }
.gradio-container main.app { padding: 0 !important; }
#hero { display: flex; justify-content: space-between; align-items: center; gap: 24px;
    padding: 6px 0 24px; border-bottom: 1px solid var(--border-color-primary); margin-bottom: 10px; }
#hero .title-line { display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
#hero h1 { font-size: 36px; line-height: 1.15; margin: 0; letter-spacing: -.04em; }
#hero .badge { color: var(--color-accent); border: 1px solid var(--border-color-primary);
    border-radius: 999px; padding: 5px 10px; font-size: 11px; font-weight: 650; letter-spacing: .04em; }
#hero p { color: var(--body-text-color-subdued); font-size: 14px; line-height: 1.6; margin: 10px 0 0; }
#hero nav { display: flex; flex-wrap: wrap; gap: 8px; flex-shrink: 0; }
#hero a { color: var(--body-text-color); text-decoration: none; border: 1px solid var(--border-color-primary);
    background: var(--block-background-fill); border-radius: 9px; padding: 8px 12px; font-size: 13px;
    transition: border-color .15s, color .15s; }
#hero a:hover { color: var(--color-accent); border-color: var(--color-accent); }
#image-panels { gap: 28px; align-items: flex-start; }
.panel-heading { min-height: 36px; align-items: center; gap: 12px; }
.section-title h3 { margin: 0 !important; font-size: 18px; font-weight: 650; letter-spacing: -.02em; }
#reset { background: transparent; border-color: transparent; min-height: 32px;
    padding: 4px 8px; color: var(--body-text-color-subdued); box-shadow: none; font-size: 12px; }
#reset:hover { color: var(--body-text-color); background: var(--background-fill-secondary); }
#source, #comparison, #restored { border: 1px solid var(--border-color-primary); border-radius: 14px;
    background: var(--block-background-fill); overflow: hidden; }
#source img { width: 100% !important; height: 100% !important; object-fit: contain; }
#result-tabs { position: relative; }
#result-tabs > .tab-wrapper { position: absolute; top: -52px; right: 0; width: 176px;
    height: 36px; border: none; padding: 0; z-index: 1; }
#result-tabs [role="tablist"] { border: 1px solid var(--border-color-primary); border-radius: 8px;
    background: var(--background-fill-secondary); padding: 3px; gap: 2px; }
#result-tabs [role="tab"] { border: none; border-radius: 5px; padding: 4px 12px;
    font-size: 12px; line-height: 20px; color: var(--body-text-color-subdued); }
#result-tabs [role="tab"][aria-selected="true"] { background: var(--block-background-fill);
    color: var(--body-text-color); box-shadow: 0 1px 3px #0f172a0d; }
#result-tabs [role="tab"]::after { display: none; }
#result-tabs > [role="tabpanel"] { padding: 0; border: none; }
#comparison > .empty { display: flex; flex-direction: column; gap: 12px; }
#comparison > .empty::after { content: "Your restored image will appear here";
    color: var(--body-text-color-subdued); font-size: 13px; }
#input-actions { align-items: flex-end; gap: 16px; padding-top: 2px; }
#size-control { --size-columns: minmax(90px, 1.25fr) 8px repeat(3, minmax(38px, 1fr));
    flex: 1 1 230px; gap: 6px; }
#size-control .html-container { padding: 0; }
.size-headings span { color: var(--body-text-color-subdued); }
.size-headings { display: grid; grid-template-columns: var(--size-columns); gap: 4px;
    color: var(--body-text-color-subdued); font-size: 12px; line-height: 18px; }
.size-headings span:last-child { grid-column: 3 / 6; }
#output-size { border: none; background: transparent; padding: 0; }
/* Keep one native radio group while visually separating the default preset from scales. */
#output-size > .wrap:not(.default) { position: relative; display: grid;
    grid-template-columns: var(--size-columns);
    grid-template-rows: 44px; gap: 4px; padding: 0; background: transparent; }
#output-size > .wrap:not(.default)::before { content: ""; grid-column: 3 / 6; grid-row: 1;
    border: 1px solid var(--border-color-primary); border-radius: 10px; pointer-events: none;
    background: color-mix(in srgb, var(--body-text-color) 3%, var(--body-background-fill)); }
#output-size .wrap > label { position: relative; grid-row: 1; min-width: 0; justify-content: center;
    margin: 4px 0; padding: 0 8px; gap: 0; border: 1px solid transparent; border-radius: 7px;
    background: transparent; box-shadow: none; font-size: 13px; font-weight: 500;
    color: color-mix(in srgb, var(--body-text-color) 76%, var(--body-background-fill));
    transition: background .15s, border-color .15s, color .15s; }
#output-size .wrap > label:nth-of-type(1) { grid-column: 1; margin: 0; border-radius: 10px;
    border-color: var(--border-color-primary); background: var(--block-background-fill);
    font-size: 14px; font-weight: 650; flex-direction: column; gap: 1px; line-height: 18px; }
#output-size .wrap > label:nth-of-type(1)::after { content: "Recommended";
    font-size: 9px; line-height: 12px; font-weight: 500; opacity: .8; }
#output-size .wrap > label:nth-of-type(2) { grid-column: 3; margin-left: 4px; }
#output-size .wrap > label:nth-of-type(3) { grid-column: 4; }
#output-size .wrap > label:nth-of-type(4) { grid-column: 5; margin-right: 4px; }
#output-size .wrap > label:hover { color: var(--body-text-color); }
#output-size .wrap > label.selected { background: var(--block-background-fill);
    color: var(--color-accent); border-color: var(--border-color-primary); box-shadow: 0 1px 3px #0f172a0a; }
#output-size .wrap > label:nth-of-type(1).selected {
    background: color-mix(in srgb, var(--color-accent) 8%, var(--block-background-fill));
    border-color: color-mix(in srgb, var(--color-accent) 38%, var(--border-color-primary));
    box-shadow: none; }
#output-size input[type="radio"] { position: absolute; inset: 0; opacity: 0;
    width: 100%; height: 100%; margin: 0; cursor: pointer; }
#output-size label:has(input:focus-visible) { outline: 2px solid var(--color-accent); outline-offset: 3px; }
#output-size label:has(input:disabled) { cursor: default; opacity: .55; }
#restore, #download { min-height: 44px; border-radius: 9px; font-size: 14px; font-weight: 600; }
#restore { box-shadow: 0 3px 8px #5148d61a; }
#download:disabled { opacity: .5; }
#result-actions { align-items: flex-end; min-height: 71px; gap: 16px; }
#compare-help { min-width: 120px; padding-bottom: 10px; }
#input-info p, #compare-help p { color: var(--body-text-color-subdued); font-size: 12px; line-height: 1.6; }
#advanced { border-width: 1px 0 0 !important; border-color: var(--border-color-primary); border-radius: 0; padding: 0;
    background: transparent; box-shadow: none; }
#advanced > .label-wrap { padding: 12px 0; font-size: 12px; color: var(--body-text-color-subdued); }
#status { padding: 10px 14px; border-radius: 9px; background: var(--background-fill-secondary);
    color: var(--body-text-color-subdued); font-size: 13px; }
#samples { padding: 8px 0 4px; }
#examples .gallery { gap: 10px; }
#examples .thumbnail-item { border-radius: 9px; }
#footnote { color: var(--body-text-color-subdued); font-size: 12px;
    border-top: 1px solid var(--border-color-primary); padding-top: 14px; margin-top: 8px; }
@media (max-width: 1100px) {
    #hero { align-items: flex-start; flex-direction: column; gap: 14px; }
}
@media (max-width: 600px) {
    .gradio-container { padding-top: 16px !important; }
    #hero h1 { font-size: 30px; }
    #hero p { font-size: 13px; }
    #image-panels { gap: 24px; }
    #source, #comparison, #restored { height: 360px !important; max-height: 360px !important; }
    #input-actions { gap: 10px; }
    #restore { flex-grow: 1 !important; }
    #result-actions { min-height: 52px; }
}
"""


def build_theme():
    return gr.themes.Base(
        primary_hue='indigo', secondary_hue='indigo', neutral_hue='slate',
        font=['Inter', 'system-ui', 'sans-serif'],
        radius_size=gr.themes.sizes.radius_lg,
    ).set(
        body_background_fill='#f8fafc',
        block_background_fill='#ffffff',
        block_border_color='#e2e8f0',
        block_shadow='none',
        block_label_background_fill='transparent',
        block_label_border_width='0px',
        block_label_text_color='*neutral_600',
        block_label_padding='0 0 6px',
        block_label_shadow='none',
        input_background_fill='#ffffff',
        button_primary_background_fill='#6259e8',
        button_primary_background_fill_hover='#5148d6',
        button_primary_border_color='#6259e8',
        button_primary_border_color_hover='#5148d6',
        button_primary_text_color='#ffffff',
        button_secondary_background_fill='#ffffff',
        button_secondary_background_fill_hover='#f1f5f9',
        button_secondary_border_color='#e2e8f0',
        button_secondary_shadow='none',
    )


def sizing_arguments(size):
    """Map a single choice to either resolution or scale, never both."""
    if size == '2K':
        return {'resolution': '2k'}
    if size in ('2×', '3×', '4×'):
        return {'upscale': float(size[0])}
    raise gr.Error('Choose 2K, 2×, 3×, or 4×.')


def input_description(image):
    if image is None:
        return 'Upload a PNG, JPEG or WebP, or choose an example below.'
    image = ImageOps.exif_transpose(image)
    return f'**Input:** {image.width:,} × {image.height:,} px · Original aspect ratio preserved'


def clear_result(image):
    return None, None, gr.update(value=None, interactive=False), READY, input_description(image)


def reset_demo():
    return (None, None, None, gr.update(value=None, interactive=False), READY,
            input_description(None), '2K', 28, 2026)


def make_generate(model, output_dir=OUTPUTS_DIR, request_validator=None):
    def generate(image, size, steps, seed,
                 progress=gr.Progress(track_tqdm=True)):
        if image is None:
            raise gr.Error('Upload an image or choose one of the examples first.')
        arguments = sizing_arguments(size)
        if steps is None or not 1 <= int(steps) <= 50:
            raise gr.Error('Sampling steps must be between 1 and 50.')
        if seed is None or not -1 <= int(seed) <= 2**32 - 1:
            raise gr.Error('Seed must be -1 (random) or an integer from 0 to 4294967295.')
        actual_seed = secrets.randbelow(2**32) if int(seed) == -1 else int(seed)
        image = ImageOps.exif_transpose(image).convert('RGB')
        if request_validator is not None:
            request_validator(image, size, int(steps))
        started = time.perf_counter()
        # Remove the preceding result before starting a new run, including on failure.
        yield None, None, gr.update(value=None, interactive=False), 'Upscaling image…'
        progress(0, desc='Preparing image')
        try:
            result = model(image, **arguments, sample_steps=int(steps), seed=actual_seed)
        except gr.Error:
            # Preserve actionable hosted-runtime errors, such as exhausted GPU quota.
            raise
        except Exception as exc:
            logging.exception('Fill2SR inference failed')
            raise gr.Error('Restoration failed. Try a smaller scale or check the server log.') from exc
        progress(0.95, desc='Preparing comparison and PNG')
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f'fill2sr_{uuid4().hex}.png'
        result.save(path, format='PNG')
        before = image.resize(result.size, Image.Resampling.NEAREST)
        elapsed = time.perf_counter() - started
        status = (f'**Complete** · {image.width:,} × {image.height:,} → '
                  f'**{result.width:,} × {result.height:,} px** · {size} · '
                  f'{elapsed:.1f}s · {int(steps)} steps · Seed {actual_seed}')
        progress(1, desc='Complete')
        yield (before, result), result, gr.update(value=str(path), interactive=True), status
    return generate


def build_demo(model, *, output_dir=OUTPUTS_DIR, request_validator=None,
               max_steps=50, delete_cache=None, usage_note=None):
    examples = [(str(EXAMPLES_DIR / name), label) for name, label in EXAMPLES
                if (EXAMPLES_DIR / name).is_file()]
    initial = examples[0][0] if examples else None
    with gr.Blocks(title='Fill2SR — ECCV 2026', theme=build_theme(), css=CSS,
                   fill_width=True, delete_cache=delete_cache) as demo:
        gr.HTML('''<header id="hero">
            <div>
                <div class="title-line"><h1>Fill2SR</h1><span class="badge">ECCV 2026 · OURS-FULL</span></div>
                <p>Repurposing Inpainting Diffusion Transformers for Real-World Super-Resolution</p>
            </div>
            <nav aria-label="Project links">
                <a href="https://doi.org/10.1007/978-3-032-37556-8_24" target="_blank" rel="noopener noreferrer">Paper ↗</a>
                <a href="https://github.com/Xingfu-Yi/Fill2SR" target="_blank" rel="noopener noreferrer">GitHub ↗</a>
                <a href="https://huggingface.co/yixingfu/Fill2SR" target="_blank" rel="noopener noreferrer">Weights ↗</a>
            </nav></header>''')
        with gr.Row(equal_height=False, elem_id='image-panels'):
            with gr.Column(scale=1, min_width=320):
                with gr.Row(elem_classes='panel-heading'):
                    gr.Markdown('### Input', elem_classes='section-title')
                    clear = gr.Button('Reset', variant='secondary', size='sm',
                                      scale=0, min_width=60, elem_id='reset')
                source = gr.Image(value=initial, label='Low-quality image', show_label=False, type='pil',
                                  sources=['upload', 'clipboard'], format='png',
                                  height=440, elem_id='source')
                with gr.Row(elem_id='input-actions'):
                    with gr.Column(scale=1, min_width=220, elem_id='size-control'):
                        gr.HTML('<div class="size-headings"><span>Resolution</span><span>Scale</span></div>')
                        size = gr.Radio(['2K', '2×', '3×', '4×'], value='2K', container=False,
                                        label='Output size', show_label=False, elem_id='output-size')
                    restore = gr.Button('Generate', variant='primary', elem_id='restore',
                                        scale=0, min_width=168)
                input_info = gr.Markdown(input_description(Image.open(initial)) if initial
                                         else input_description(None), elem_id='input-info')
                with gr.Accordion('Advanced settings', open=False, elem_id='advanced'):
                    steps = gr.Slider(1, max_steps, value=28, step=1, label='Sampling steps')
                    with gr.Row():
                        seed = gr.Number(value=2026, minimum=-1, maximum=2**32 - 1, precision=0,
                                         label='Seed', info='Use -1 for a random seed.', min_width=160)
                        random_seed = gr.Button('Randomize seed', size='sm', scale=0, min_width=120)
                    gr.Markdown('2K uses the model’s resolution preset. '
                                '2× / 3× / 4× multiply the input dimensions.')
            with gr.Column(scale=1, min_width=320):
                with gr.Row(elem_classes='panel-heading'):
                    gr.Markdown('### Result', elem_classes='section-title')
                with gr.Tabs(elem_id='result-tabs'):
                    with gr.Tab('Compare'):
                        comparison = gr.ImageSlider(
                            label='Input / Fill2SR', show_label=False, type='pil', format='png',
                            height=440, max_height=440, interactive=False,
                            show_download_button=False, elem_id='comparison')
                    with gr.Tab('Output'):
                        output = gr.Image(label='Restored image', show_label=False, type='pil', format='png',
                                          height=440, interactive=False, elem_id='restored')
                with gr.Row(elem_id='result-actions'):
                    gr.Markdown('Drag to compare before and after.', elem_id='compare-help')
                    download = gr.DownloadButton('Download PNG', interactive=False,
                                                  scale=0, min_width=152, elem_id='download')
        status = gr.Markdown(READY, elem_id='status')
        if usage_note:
            gr.Markdown(usage_note)
        example_picker = None
        if examples:
            with gr.Column(elem_id='samples'):
                gr.Markdown('### Try an example', elem_classes='section-title')
                example_picker = gr.Examples(examples=[[path] for path, _ in examples], inputs=source,
                            examples_per_page=len(examples), label='RealLQ250 · paper examples · click an image to load', elem_id='examples',
                            cache_examples=False)
        gr.Markdown('Research demo · [Paper](https://doi.org/10.1007/978-3-032-37556-8_24) · '
                    '[RealLQ250](https://github.com/shallowdream204/DreamClear)\n\n'
                    'The model is subject to the FLUX.1 [dev] non-commercial license. '
                    '[Model license and attribution](https://huggingface.co/yixingfu/Fill2SR).',
                    elem_id='footnote')
        result_outputs = [comparison, output, download, status]
        source.change(clear_result, source, result_outputs + [input_info], queue=False,
                      api_name=False)
        random_seed.click(lambda: secrets.randbelow(2**32), outputs=seed,
                          queue=False, api_name=False)
        controls = [source, size, steps, seed, clear, restore, random_seed]
        if example_picker is not None:
            controls.append(example_picker.dataset)
        # Keep the displayed input and settings tied to the active request.
        def set_busy(busy):
            return [gr.update(visible=not busy) if isinstance(control, gr.Dataset)
                    else gr.update(interactive=not busy) for control in controls]

        restore.click(lambda: set_busy(True),
                      outputs=controls, queue=False, api_name=False).then(
            make_generate(model, output_dir, request_validator), [source, size, steps, seed],
            result_outputs, api_name='generate', concurrency_limit=1,
            concurrency_id='fill2sr-gpu', trigger_mode='once',
        ).then(lambda: set_busy(False),
               outputs=controls, queue=False, api_name=False)
        clear.click(reset_demo, outputs=[source, comparison, output, download, status, input_info,
                                        size, steps, seed],
                    queue=False, api_name=False)
    return demo.queue(default_concurrency_limit=1, max_size=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--server-name', default='127.0.0.1')
    parser.add_argument('--server-port', type=int, default=7860)
    parser.add_argument('--fill-dir', type=Path)
    parser.add_argument('--redux-dir', type=Path)
    parser.add_argument('--lora', type=Path)
    args = parser.parse_args()
    from fill2sr import Fill2SRPipeline
    fill, redux, lora = checkpoint_paths(args.fill_dir, args.redux_dir, args.lora)
    model = Fill2SRPipeline(fill, redux, lora, off_load=args.offload, device=args.device)
    demo = build_demo(model)
    demo.launch(server_name=args.server_name, server_port=args.server_port,
                allowed_paths=[str(EXAMPLES_DIR), str(OUTPUTS_DIR)])


if __name__ == '__main__':
    main()
