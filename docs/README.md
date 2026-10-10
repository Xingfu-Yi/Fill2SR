# Fill2SR project website

Static HTML, CSS and JavaScript. No build step, external JavaScript, analytics,
GPU backend or model weights are needed. The intended GitHub Pages address is
<https://xingfu-yi.github.io/Fill2SR/>.

## Local preview

From the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open <http://127.0.0.1:8000/>. Use an HTTP server, not `file://`, because the
viewer loads `gallery.json` and an ES module. All runtime asset URLs are relative,
so the same files work below the `/Fill2SR/` project path.

The site displays **precomputed outputs**. New-image inference is provided by the
repository's local Gradio app. Publishing this directory is a separate release
step; editing or previewing it does not deploy a website.

## Showcase provenance

The 16 inputs match every example row in Figs. A3–A6 of the arXiv appendix.
Nokia is first, followed by the remaining inputs in paper order. This selection
is shared with Gradio through `assets/examples/reallq250/selection.json`.
All interactive outputs use Ours-Full, the 2K preset, 28 steps and seed 2026.
The Nokia and microchip outputs reuse verified earlier runs with the same
checkpoint and settings; the other 14 outputs were generated for this selection.
All inputs are unmodified 256 × 256 PNGs from RealLQ250; all outputs are unmodified
1,536 × 1,536 model PNGs. The paper's 4× results remain in the original figures;
they are not reused or relabeled as 2K interactive results.

`gallery.json` records the selected IDs, source, model hash, inference code
revision per example, figure/row references, sampling settings, dimensions and
hashes for both images of each pair.
The 2K preset follows the model's pixel budget, not a fixed 2,048-pixel edge.
Thumbnails are 240 × 240 WebP derivatives of the outputs. The viewer loads only
the active full-resolution pair; original input and output PNGs can be downloaded.

The promotional figure `assets/nokia-comparison.jpg` uses the full image on both sides.
Its matching detail region is `[36, 118, 103, 153]` in input pixel coordinates
(left, top, right, bottom, right/bottom excluded), and six times those coordinates
in the output. Inputs are enlarged with nearest-neighbor interpolation; outputs
are reduced with Lanczos for display. No model-output pixels were retouched.

Only these 16 pairs are included in the website. Full generation records and
the earlier 32-candidate exploration are retained separately in the maintainer's
ignored output directories. This qualitative display does not replace quantitative
evaluation of the full set.

RealLQ250 was released with [DreamClear](https://github.com/shallowdream204/DreamClear).
Third-party images retain their original rights; the repository's source-code
license does not relicense them. Outputs are generated restorations, not ground
truth. The model's license and attribution are in its
[Hugging Face repository](https://huggingface.co/yixingfu/Fill2SR).

## Paper figures

The teaser, real-world and synthetic comparisons, all five RealLQ250 appendix comparison
figures, 12K artwork and RCDT illustration are included (10 figures in total).
The identity study, boundary comparison and ablation figure remain available in
the repository and README; they are omitted from the project website. The original
PNGs in `assets/` remain unchanged. `paper-figures.json` records the source hashes
and display derivatives for the figures shown on the website.

Run `python scripts/prepare_paper_figures.py` from the repository root to regenerate
the uncropped previews (Pillow required). They are resized to at most 2400 pixels
wide and saved as quality-94 WebP files. All figures retain their complete frames,
labels and crops. The website links to the original PNGs and available PDFs.
These paper experiments use the settings labeled in the figures, independently
of the interactive gallery's uniform 2K configuration.

The comparison section directly displays seven complete figures: the main real-world
comparison, synthetic Fig. A8, and all five RealLQ250 figures A3–A7 (20 example
rows). The RCDT data-synthesis diagram (Fig. A2) sits beside Fig. A7 as an
equally wide card in the final row. There are no tabs or hidden comparison panels.
Desktop uses two columns; mobile stacks the cards in one column. Every figure opens at full window width, preserving its aspect ratio and allowing
vertical scrolling. Whole image fits the entire figure. Click or tap the enlarged
image again to close it and return to the original card; scrolling and dragging
do not dismiss it. The toolbar stays visible, and Close or Escape also dismisses
the viewer. All appendix figures A1–A8 are represented, including the
12K artwork in its own section and the RCDT illustration beside the comparisons.

The three new PNGs (`real_world_examples_4`, `real_world_examples_5`, and
`rcdt_pipeline`) are full-page renders of the corresponding one-page figure PDFs,
using Poppler: `pdftoppm -singlefile -scale-to-x 2400 -scale-to-y -1 -png input.pdf output`.
The figure manifest records the matching PDF hashes and appendix figure numbers.
Figures outside the first screen load lazily. Without JavaScript, the figures
remain visible.

## Checks

```bash
python -m unittest discover -s tests -v
node --test tests/viewer.test.mjs
python scripts/check_showcase.py
```

Browser checks should include desktop and narrow-screen layouts, divider dragging
and keyboard control, zoom, image pan, fit, 100%, reset, fullscreen, rapid example
switching and both PNG downloads. Also check all seven visible comparison cards,
appendix figure labels, full-width opening, vertical scrolling, Whole image, click-to-close, Escape and original-asset links. Test physical two-finger pinch on a touchscreen
before release; geometry tests cover the pinch midpoint and scale calculations.

## Publish after review

The release owner must explicitly approve public release first. Once approved,
push the reviewed commit, configure GitHub Pages to deploy from the `main` branch
and `/docs` directory, and check the deployed project URL. The `.nojekyll` file
makes this an ordinary static site. No automatic deployment workflow is included.
Coordinate this step with the model repository's public, automatically approved
gated release so that the code, model and website links resolve together.
