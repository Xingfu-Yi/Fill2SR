# RealLQ250 example inputs

These 16 unmodified images come from the RealLQ250 benchmark released with
[DreamClear](https://github.com/shallowdream204/DreamClear). They match all 16
example rows in Figs. A3–A6 of the Fill2SR arXiv appendix. Nokia remains the
first/default example; the remaining images follow their order in the paper.
All inputs are 256 × 256 pixels, with no extra degradation or resizing.

The Gradio demo and project website use this same selection, recorded with input
hashes and figure/row references in [selection.json](selection.json). The website
outputs use Ours-Full, the 2K preset, 28 steps and seed 2026. The appendix figures
retain their original 4× experiment settings; their outputs are not reused as
2K results.

| Original filename | Example label | Paper reference |
| --- | --- | --- |
| 113.png | Nokia phone (default) | Fig. A5, row 4 |
| 001.png | Warrior portrait | Fig. A3, row 1 |
| 005.png | Microchip lettering | Fig. A3, row 2 |
| 032.png | Wooden walkway | Fig. A3, row 3 |
| 041.png | Palace architecture | Fig. A3, row 4 |
| 050.png | City buildings | Fig. A4, row 1 |
| 076.png | Archival street scene | Fig. A4, row 2 |
| 077.png | Aircraft lettering | Fig. A4, row 3 |
| 079.png | Orange SUV | Fig. A4, row 4 |
| 088.png | Bus-route display | Fig. A5, row 1 |
| 094.png | Panda | Fig. A5, row 2 |
| 097.png | Anime portrait | Fig. A5, row 3 |
| 120.png | Archival portrait | Fig. A6, row 1 |
| 137.png | Cat | Fig. A6, row 2 |
| 144.png | Bicycle by the beach | Fig. A6, row 3 |
| 197.png | Illustrated street | Fig. A6, row 4 |

The images retain the rights of their respective owners; the repository's code
license does not relicense third-party image content. Refer to the source benchmark
for dataset information and attribution.
