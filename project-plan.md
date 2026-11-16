# Project Plan: Quantitative Reasoning Benchmark for VLMs on Pathology Images

## Goal
Build a contamination-resistant benchmark of auto-checkable quantitative questions on real
pathology images. Use it to (1) measure how reliable frontier vision-language models (VLMs) are,
(2) diagnose whether failures come from perception or reasoning, and (3) test whether tool use
or RL post-training closes the gap.

Ground truth comes from existing expert annotation masks (PanNuke, MoNuSeg). Answers are
computed in code from the masks; no manual labeling. Manual work is limited to spot-checking.

## Constraints
- Limited compute: laptop + free Colab. API budget ~$50–150. Optional rented A100 (<$50) for training.
- Python 3.11, PyTorch, scikit-image, PIL, numpy, pandas. Keep dependencies minimal.
- Check dataset licenses before redistributing images; if restricted, release generation code + image IDs only.

## Repo layout
```
vlm-bioimage-bench/
├── data/raw/            # downloaded datasets (git-ignored)
├── bench/
│   ├── sources/         # loaders → common Sample/Nucleus dataclasses
│   ├── templates/       # one module per question template
│   ├── render.py        # boxes, numbered labels, quadrant grid, upscaling
│   └── build.py         # writes benchmark.jsonl + rendered images
├── eval/
│   ├── models.py        # common Model interface (API + open models)
│   ├── run.py           # cached, retrying, concurrent runner
│   └── score.py         # answer parsing + scoring
├── analysis/            # bootstrap CIs, difficulty curves, ablations, figures
├── train/               # stretch: LoRA SFT / GRPO
└── tests/
```

## Core data structures
```python
@dataclass
class Nucleus:
    centroid: tuple[float, float]   # (row, col)
    area: int
    bbox: tuple[int, int, int, int]
    eccentricity: float
    ntype: str | None               # e.g. "neoplastic", "inflammatory"

@dataclass
class Sample:
    image: np.ndarray               # H x W x 3 uint8
    nuclei: list[Nucleus]
    meta: dict                      # dataset, tissue, fold, idx
```

Each benchmark item is one JSONL line:
```json
{"id": "count_in_box_000417", "template": "count_in_box", "images": ["img/000417.png"],
 "prompt": "... End with 'Answer: <integer>'.", "answer": 23, "answer_type": "int",
 "difficulty": {"true_count": 23}, "source": {"dataset": "pannuke", "tissue": "Colon", "idx": 1180}}
```

## Question templates (v1)
| Template | Question | Difficulty knob | Answer type |
|---|---|---|---|
| `count_in_box` | Nuclei centered in red box? | true count (3–60) | int |
| `count_type` | How many inflammatory cells? | count, type mix | int |
| `compare_density` | Patch A or B has more nuclei? | count ratio (2.0 → 1.1) | A/B |
| `largest_marked` | Which numbered nucleus is largest? | area gap (≥25%) | label |
| `locate_quadrant` | Which quadrant has most nuclei? | top-2 gap | choice |
| `tumor_fraction` | % tumor cells in box (nearest 10%) | count, mix | int (percent) |

Ambiguity filters (required): skip boxes with any centroid within `margin` px of an edge;
skip comparisons below the minimum gap; upscale 256px patches to ~768px before rendering.

Control variants for each item:
- `image` (default)
- `text_only`: nucleus coordinates/types as text, no image
- `tools` (phase 4): image + `crop`, `zoom`, `segment` tools

## Phases

### Phase 1 — Data + generator (week 1–2)
- [ ] PanNuke loader → `Sample` via `skimage.measure.regionprops`
- [ ] MoNuSeg loader
- [ ] Renderer (box, numbered labels, quadrant grid, side-by-side A/B)
- [ ] All v1 templates with ambiguity filters + unit tests
- [ ] `build.py` → ~1,500 items, stratified by template, tissue, difficulty; fixed seed
- [ ] Spot-check tool: shows image + prompt, records my answer, compares to ground truth
- **Done when:** my spot-check accuracy ≥90% per template on 50 items

### Phase 2 — Eval harness (week 3)
- [ ] `Model` protocol: `generate(prompt, images, **kw) -> str`
- [ ] API wrappers + one open VLM wrapper (HF transformers, 4-bit on Colab)
- [ ] Cache responses keyed by hash(model, prompt, image hashes, params) → `results/{model}.jsonl`
- [ ] Retries with backoff, bounded concurrency, temperature 0
- [ ] Strict parser (last `Answer: ...` match); track parse-failure rate
- [ ] Pilot: 100 items × 2 models; log cost per item
- **Done when:** pilot runs end-to-end and cost estimate for full run is known

### Phase 3 — Full eval + baselines + analysis (week 4–5)
- [ ] 4–6 models × {direct, chain-of-thought} × {image, text_only}
- [ ] Baselines: random/majority, median-count, classical CV (threshold + watershed, or Cellpose)
- [ ] Confidence (0–100) elicitation on a subset → calibration/ECE
- [ ] Main table with 95% bootstrap CIs
- [ ] Accuracy vs. difficulty curves; predicted-vs-true count scatter
- [ ] Perception-vs-reasoning plot (image vs. text_only)
- [ ] 10–15 annotated failure examples
- **Done when:** all figures reproducible from `analysis/` with one command

### Phase 4 — Stretch: tools and/or training (week 6+)
- [ ] Tool-use variant (`crop`, `zoom`, `segment`) and compare to image-only
- [ ] Generalization splits: held-out tissues, held-out templates
- [ ] LoRA SFT on ~3B open VLM; then GRPO with `score()` as reward (check current TRL/verl VLM support)
- [ ] Report in-distribution vs. held-out gains and any reward hacking

### Phase 5 — Release
- [ ] README with results table + difficulty-curve figure; `make bench`, `make eval MODEL=...`
- [ ] Hugging Face dataset card (templates, filters, licenses, limitations)
- [ ] 4–6 page workshop-style writeup

## Conventions
- Everything deterministic from a seed; record seed + dataset version in item metadata.
- Never call a paid API without the cache layer; run pilots before full runs.
- Tests for every template: answer matches a hand-computed toy mask.
- Don't use any lab (Tearney/Mahmood) data.

## First task for Claude Code
Scaffold the repo, implement the PanNuke loader, `render.py`, and the `count_in_box` and
`compare_density` templates with tests, then build a 50-item sample and save rendered images
for me to review.