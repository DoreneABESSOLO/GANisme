# GAN — Generating Paintings from the Web Gallery of Art Catalogue

A data science project: build, starting from the **Web Gallery of Art** (WGA —
European art from the 3rd to the 19th century) catalogue, an exploitable image
dataset to train a **GAN (DCGAN)** capable of generating original paintings.

The project follows a three-stage pipeline, each documented by an end-to-end
executable Jupyter notebook:

```
WGA catalogue  →  EDA & selection  →  Image scraping  →  Preprocessing  →  GAN training
(52,867 works)     (32,438 paintings)  (3,131 images)      (crop/normalization)  (DCGAN, BCE)
```

A presentation deck (`presentation/gan_peintures.pptx`) summarizes the whole
project with the visualizations produced by the notebooks.

## Table of contents

- [Data source](#data-source)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Exploratory data analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Fetching the images (scraping)](#2-fetching-the-images-scraping)
  - [3. Preprocessing](#3-preprocessing)
  - [4. Training the GAN](#4-training-the-gan)
- [Current results](#current-results)
- [Methodological choices](#methodological-choices)
- [Known limitations & next steps](#known-limitations--next-steps)
- [Legal and ethical considerations](#legal-and-ethical-considerations)

## Data source

The catalogue comes from the [Web Gallery of Art](https://www.wga.hu), a
non-commercial database maintained by two people, covering European art from
the 3rd to the 19th century (painting, sculpture, architecture, graphic
arts…). It is provided as a downloadable file (`data/art_catalog.xlsx` /
`catalog.csv`) and contains **52,867 works** described by 11 text fields:
`AUTHOR`, `BORN-DIED`, `TITLE`, `DATE`, `TECHNIQUE`, `LOCATION`, `URL`,
`FORM`, `TYPE`, `SCHOOL`, `TIMEFRAME`.

The catalogue contains **no images**: only URLs to HTML presentation pages on
wga.hu are provided — fetching the images is an integral part of this project
(see [step 2](#2-fetching-the-images-scraping)).

## Project structure

```
GAN/
├── README.md                      # this file
│
├── eda_peintures.ipynb            # exploratory data analysis (EDA) notebook
├── eda_peintures.py                # equivalent non-interactive script
├── report_data.json                # aggregates produced by the EDA
│
├── scraping_images.ipynb           # local scraping (VS Code / Jupyter)
├── scraping_images_colab.ipynb     # self-contained Google Colab variant
├── scrape_images.py                # download engine (shared module)
├── build_subset.py                 # stratified sampling (shared module)
│
├── preprocessing_images.ipynb      # preprocessing pipeline + visualizations
│
├── gan_dataset.py                   # shared PyTorch Dataset (filtering, crop, normalization)
├── gan_models.py                    # DCGAN architecture (Generator / Discriminator)
├── train_gan.ipynb                  # GAN training notebook
│
├── data/
│   ├── art_catalog.xlsx             # source catalogue (downloaded from wga.hu)
│   ├── art_catalog.xlsx - catalog.csv
│   ├── paintings_only.csv           # FORM == "painting" subset (32,438 rows)
│   └── subset_sample.csv            # stratified sample (3,680 rows targeted)
│
├── images/                          # downloaded images (mirrors the wga.hu URL structure)
│   └── _failed.csv                  # download-failure log
│
├── checkpoints/                     # saved GAN weights (generator.pt, discriminator.pt)
│
└── presentation/
    └── gan_peintures.pptx           # project presentation deck
```

> `EDA.ipynb` (at the root, uppercase) is an earlier, unrelated draft — ignore
> it, kept as-is without modification.

## Installation

Python ≥ 3.10 recommended. Main dependencies:

```bash
pip install pandas numpy matplotlib pillow requests
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU build
pip install nbformat nbclient ipykernel      # to run the notebooks from the command line
```

No API key is required. The notebooks run equally well locally (VS Code,
Jupyter) or on Google Colab (see `scraping_images_colab.ipynb` and the
"Colab compatibility" section of `train_gan.ipynb`).

## Usage

### 1. Exploratory data analysis (EDA)

```bash
python eda_peintures.py
# or, as a notebook: open and run eda_peintures.ipynb
```

Produces `data/paintings_only.csv` (paintings subset) and `report_data.json`
(aggregates for the presentation). Covers: data quality, breakdown by
genre/school/period, artists, technique and dimensions, reliability of the
`DATE` field, and conservation locations.

**Main finding:** the catalogue is clean at the structural level (no
duplicates, no empty fields) but heavily imbalanced — 42% religious painting,
42% Italian school, no works after 1900.

### 2. Fetching the images (scraping)

The mapping between a presentation page and its full-resolution image is
systematic:

```
https://www.wga.hu/html/{a}/{artist}/{work}.html
    → https://www.wga.hu/art/{a}/{artist}/{work}.jpg
```

Build a stratified sample (equal quota by genre, then by school, to correct
the imbalance found in the EDA):

```bash
python build_subset.py --size 4000
```

Then download the images (deliberately rate-limited — see
[legal considerations](#legal-and-ethical-considerations)):

```bash
python scrape_images.py --csv data/subset_sample.csv --out images --workers 4 --min-delay 0.4
```

Useful options: `--limit N` (test on a sample), `--workers`, `--min-delay`,
`--timeout`. The script automatically resumes where it left off (files
already present are skipped) and logs failures to `images/_failed.csv`.

To run the scraping from Google Colab instead of locally, use
`scraping_images_colab.ipynb` (self-contained: it rewrites its own
dependencies via `%%writefile`, and mounts Google Drive to persist images
across sessions).

### 3. Preprocessing

`preprocessing_images.ipynb` documents and visualizes each step on the
actually downloaded images:

1. File validation (detecting corrupted downloads)
2. Analysis of real dimensions and aspect ratios
3. Filtering out extreme formats (width/height ratio outside 0.4–2.5, or
   shortest side < 64 px)
4. Center-crop vs. letterbox comparison (center-crop retained)
5. Normalization (`RGB → crop/resize 64×64 or 128×128 → tensor → [-1, 1]`)
6. Robust PyTorch `Dataset` / `DataLoader` (skips an unreadable image rather
   than crashing)
7. Optional augmentation (horizontal flip, with a caveat about pictorial
   conventions)

The reusable logic (dataset, filtering, center-crop) is factored out into
`gan_dataset.py`, imported by both this notebook and `train_gan.ipynb`.

### 4. Training the GAN

```bash
# open and run train_gan.ipynb (local or Colab)
```

Key points of the notebook:

- **Data**: prioritizes `data/subset_sample.csv` (stratified sample) over an
  arbitrary mix of everything in `images/`, via
  `gan_dataset.subset_image_paths()`.
- **Automatic resume**: if `checkpoints/generator.pt` already exists, the
  weights and loss history are reloaded, and training resumes at the next
  epoch — useful for a long CPU training run split across several sessions.
- **Periodic checkpoints** (`CKPT_EVERY`, 25 epochs by default): limits
  progress loss in case of interruption.
- **Progress visualization**: a sample generated from a fixed latent noise
  vector is displayed regularly, to visually track the generator's
  convergence.

On CPU, a full training run (several hundred epochs on a few thousand
images) can take several hours; `train_gan.ipynb` includes a "Colab
compatibility" section to migrate the execution — not the scraping — to
Colab's free GPU if needed.

## Current results

| | |
|---|---|
| Works in the catalogue | 52,867 |
| Selected paintings (`FORM == "painting"`) | 32,438 (61.4%) |
| Targeted stratified sample | 3,680 images |
| Images actually downloaded | 2,973 succeeded / 3,680 (81%), 3,131 available in total |
| Saved GAN checkpoints | `checkpoints/generator.pt`, `checkpoints/discriminator.pt` |

A first validation run (DCGAN, 30 epochs, 154 images, local CPU) confirmed
that the pipeline trains correctly: the discriminator's loss stays stable
(never collapses to zero), the generator's loss oscillates without
exploding, and the generated samples visibly evolve from pure noise toward
coherent textures and colors — without yet producing recognizable shapes,
which is expected at this scale of data/epochs. Additional runs have since
been performed locally on a larger image set; detailed results from those
runs are not yet consolidated in this repository.

The detail (loss curves, progression grids, real-vs-generated comparison) is
visualized in `train_gan.ipynb` and reused in
`presentation/gan_peintures.pptx`.

## Methodological choices

- **Architecture: DCGAN** rather than StyleGAN or a diffusion model — the
  only realistic architecture to train CPU-only on a corpus of a few
  thousand images; well documented, a standard starting point.
- **Loss function: non-saturating BCE** (the generator maximizes
  `log D(G(z))` instead of minimizing `log(1-D(G(z)))`, to avoid near-zero
  gradients early in training) + **label smoothing** (0.9 on the real side)
  to limit discriminator overconfidence, a known mode-collapse factor. Adam
  hyperparameters (`lr=2e-4`, `β=(0.5, 0.999)`) recommended by the original
  DCGAN paper.
- **Center-crop rather than letterbox** for image formatting: since extreme
  formats are already filtered out upstream, center-crop loses little real
  content and avoids introducing artificial borders the model could
  wrongly learn.
- **Stratified sampling** (equal quota by genre then by school) to mitigate
  the catalogue's structural imbalance rather than ignoring it.

## Known limitations & next steps

- **Current scale is limited**: the documented results come from a
  validation run (154 images, 30 epochs) that proves the pipeline's
  mechanics but does not yet produce convincing paintings. A full training
  run requires several thousand images and several hundred epochs.
- **Corpus imbalance**: even stratified, the sample remains dominated by
  the Italian school and the religious genre. A conditional GAN (on `TYPE`
  and/or `SCHOOL`) is a path to better control the generated diversity.
- **~18% download failures** (timeouts) on the full scraping of the
  stratified sample — due to the WGA server's limited capacity, not a
  block; re-running `scrape_images.py` with the same `--csv` automatically
  completes the missing files.
- **Reliability of the `DATE` field**: very heterogeneous free text (only
  32% exact years) — systematically prefer `TIMEFRAME` for any temporal
  filtering or conditioning.
- **Mode collapse to monitor** at larger scale: if the loss curves drift,
  switching to a **hinge loss** or an **LSGAN** is a localized change (one
  cell in `train_gan.ipynb`), not a rewrite.

## Legal and ethical considerations

The Web Gallery of Art states (`wga.hu/legal.html`) that images downloaded
from the site are intended for **educational and personal use** only; any
redistribution is prohibited without authorization. As a result:

- `scrape_images.py` deliberately limits the rate and concurrency of
  requests (non-commercial site, maintained by two people);
- the downloaded images (`images/`) and trained weights (`checkpoints/`)
  are intended for local experimentation — do **not** republish or
  redistribute them as-is;
- most of the reproduced works are in the public domain (works predating
  1900), but the *photographic reproductions* themselves remain subject to
  the usage terms of the site that hosts them.
