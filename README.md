# AI-Text Detection Neurons in Frozen BERT

Code and result data for **"A Mechanistic Study of AI-Text Detection Neurons in Frozen BERT: Sparse Probing and Activation Patching on RAID"** (EMNLP 2026).

We apply an L1→L2 sparse-probing protocol to all 9,216 CLS hidden-state dimensions of a **frozen** BERT-base-uncased (12 layers × 768, no fine-tuning) and recover a stable set of **under 1% of neurons** per generator. Bidirectional activation patching shows that set flips predictions an order of magnitude more often than size-matched random sets; mean ablation shows it is largely *not* necessary. Across generators, instruction-tuned models concentrate their stable neurons in layer 12 while base models do not.

## Paper

```bibtex
@inproceedings{blicharz-grunwald-2026-detection-neurons,
  title     = {A Mechanistic Study of {AI}-Text Detection Neurons in Frozen {BERT}:
               Sparse Probing and Activation Patching on {RAID}},
  author    = {Blicharz, Pawe{\l} and Grunwald, Mi{\l}osz},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural
               Language Processing},
  year      = {2026}
}
```

Paper source is in [`_paper/`](_paper/) (`main.tex`, `refs.bib`, `main.pdf`).

## Quickstart: figures without any download

The repository ships the result summary JSONs, so **all four paper figures regenerate in under 15 seconds from a fresh clone**, with no dataset and no GPU:

```bash
uv sync
uv run _paper/gen_figures.py
```

This writes `_paper/figures/{fig_flip_rate,fig_layer_dist,fig_jaccard,fig_logo}.pdf` and prints the underlying numbers (flip-rate ratios, layer distributions, the 6×6 Jaccard matrix, LOGO retention). This is the only path that works without the dataset — everything below requires the full pipeline.

## Install

Requires Python ≥ 3.10; `uv sync` fetches 3.11 automatically if your system Python is older. `uv.lock` pins exact versions of all 117 resolved packages. `requirements.txt` carries the same direct dependencies as ranges for `pip` users.

```bash
uv sync
```

Verify the install — this catches an incomplete environment before you commit to a long run:

```bash
uv run python -m compileall -q raid_pipeline raid_analysis scripts extra_analyses
uv run python -c "import raid_pipeline, raid_analysis; print('ok')"
```

**GPU (optional, speeds up activation extraction only).** `uv.lock` pins the CPU build of `torch` from PyPI, so install the CUDA build *after* syncing, not before:

```bash
uv sync
uv pip install torch --torch-backend=cu124
```

Note that a later `uv sync` restores the locked CPU build — re-run the second command if that happens. Every stage except activation extraction is CPU-bound, so this is optional.

## Data

RAID is **not redistributed here** — the download script fetches it. RAID is MIT-licensed and published by [Dugan et al. (2024)](https://aclanthology.org/2024.acl-long.674/).

```bash
uv run scripts/download_raid.py
```

This streams the ~12 GB RAID train CSV from HuggingFace and keeps only clean (non-adversarial) rows, writing per-(domain, model) CSVs to `data/raw/raid/`. No token is needed — the dataset is public. Set `HF_TOKEN` in a `.env` file (see `.env.example`) only if you hit rate limits.

**Budget tens of minutes to a few hours** for this step, depending on your
connection - it streams the whole 12 GB to keep ~660 MB. It is the longest
single step in the pipeline and is not in the runtime table below, which
starts after the data is on disk.

**The download is not resumable and leaves no completion marker.** It writes
CSVs incrementally, so an interrupted run leaves truncated files that later
steps will consume as if they were complete, silently corrupting your results.
The only success signal is a `DOWNLOAD COMPLETE` line at the end. If a run is
interrupted for any reason, delete the directory and start again:

```bash
rm -rf data/raw/raid && uv run scripts/download_raid.py
```

Then tokenize and cache activations. **Both steps run once per generator** - repeat for each of the six (or all eleven, for the LOGO experiment):

```bash
uv run scripts/tokenize_raid.py --model gpt4
uv run scripts/extract_activations.py \
    --tokenized-path data/processed/raid_gpt4 \
    --output results/activations_raid_gpt4
```

Substitute the generator in all three places - `--model <G>`,
`data/processed/raid_<G>`, `results/activations_raid_<G>` - using underscores
in directory names where the generator has a hyphen (`mistral-chat` ->
`raid_mistral_chat`).

### Disk budget

The streamed CSV is ~12 GB, but far less lands on disk:

| What | Size |
|---|---|
| `data/raw/raid/` (clean rows, all 11 generators) | ~660 MB |
| `data/processed/` (tokenized, all 11 generators) | ~2.1 GB |
| `results/activations_raid_<G>/` | ~350 MB per generator |
| Experiment outputs (`.npy` probe weights, splits) | ~110 MB per full run |

Budget roughly **8 GB** for the six main-body generators, or ~13 GB for all eleven (needed for the LOGO experiment).

## Reproducing the paper

Six generators carry the main results: `gpt4 gpt2 mpt mistral-chat llama-chat cohere-chat`. The LOGO experiment (Table 4) additionally needs all eleven, since it holds out whole model families.

`sparse_probe` must run before anything that depends on it. `run_all.py` resolves that ordering for you:

```bash
uv run scripts/experiments/run_all.py \
    --generators gpt4 gpt2 mpt mistral-chat llama-chat cohere-chat
```

This is a multi-hour job (see Runtime below). It prints a `run_id` at the start — **keep it**; the section after the table explains why.

| Paper | What | Command |
|---|---|---|
| **Table 1** | Sparse-probe results, 6 generators | `uv run scripts/experiments/run_experiment.py sparse_probe --generator <G>` |
| **Table 2**, **Fig 2** | Forward flip rates; flip rate vs *k* | `uv run scripts/experiments/run_experiment.py patching --generator <G>` |
| **Table 3** | Mean-ablation accuracy drop | `uv run scripts/experiments/run_experiment.py ablation --generator <G>` |
| **Fig 3**, **§6.2**, **Table 9**, **Table 11**, **Fig 5**, **Table 12** | Layer distribution, Jaccard overlap, CAV diagnostics | `uv run scripts/experiments/run_experiment.py characterize` |
| **Fig 4**, **Table 4** | Leave-one-family-out generalisation | `uv run scripts/experiments/run_logo.py` |
| **Table 5** (App B) | Stability sweep over (C, N) | `uv run scripts/experiments/run_sample_size_stability.py` |
| **Table 6** (App C) | Per-cell L2-probe metrics | `sparse_probe` output at `results/experiments/<run_id>/sparse_probe/<G>/seed_*/fold_*/eval_metrics.json` |
| **Table 7** (App D) | AUC-ROC vs L1 selection | `uv run scripts/experiments/run_experiment.py auc_comparison --generator <G>` |
| **Table 8** (App E) | Restricted probe | `uv run scripts/experiments/run_experiment.py restricted_probe --generator <G>` |
| **Table 10** (App G) | Per-domain patching breakdown | `patching` output at `results/experiments/<run_id>/patching/<G>/seed_*/fold_*/eval_metrics.json` |
| **Table 13** (App J) | Full bidirectional *k*-sweep | `patching` and `patching_human_into_ai` output, same paths |
| **Table 14** (App K) | Neuron–surface-feature association | `uv run extra_analyses/experiments/exp_08_neuron_feature_correlation.py` |
| **App L** | Worked patching example | `uv run extra_analyses/experiments/exp_02_worked_example.py` |

`uv run scripts/experiments/run_experiment.py --list` shows the available experiment names.

### Run IDs, and a gotcha worth knowing

Each invocation writes to `results/experiments/<run_id>/`, auto-generated from a timestamp unless you pass `--run-id`. Dependent experiments find their upstream run via `--source-dir` (or `source_experiment` in the YAML config).

**Three scripts hardcode the run directories from the published run** — `_paper/gen_figures.py` (lines 12–13) and both `extra_analyses/experiments/exp_*.py`. They read `20260510_rerun_main` and `20260513_190517`, which is why the shipped figures regenerate out of the box. If you re-run the pipeline yourself, either pass `--run-id 20260510_rerun_main` so your output lands where those scripts look, or edit the paths at the top of each file.

The two `extra_analyses` scripts additionally need `.npy` intermediates (probe weights, cached activations) from a completed `sparse_probe` and `patching` run. Those are **not** shipped — only the summary JSONs are — so both scripts fail with a bare `FileNotFoundError` on a fresh clone. That is expected, not a bug.

Configuration lives in `config/experiments/*.yaml`: 5 folds × 3 seeds (42, 123, 456), `max_samples: 7500`, `C=0.005`, stability threshold 0.8. The choice of `C` is itself justified by Table 5's sweep.

## Runtime and hardware

From the paper's reported budget, on a laptop (AMD Ryzen 5 5600H, RTX 3060 Laptop, 6 GB VRAM):

| Stage | Time | Bound by |
|---|---|---|
| Activation extraction, 45,000 samples | ~30 min | GPU |
| Six per-generator main runs, incl. *k*-sweep | ~5 h | CPU |
| Multi-generator stability grid | ~1 h | CPU |
| LOGO across five family folds | ~15 min | CPU |
| **Full pipeline** | **~7–8 h** | |

BERT-base (110M) runs inference-only throughout — nothing is fine-tuned. After the initial encoding pass everything operates on cached representations and is CPU-bound, so a GPU helps only the first stage.

## Layout

```
raid_pipeline/      RAID loading, tokenization, BERT activation extraction
raid_analysis/      Library: data, selection (L1/AUC), evaluation (probe,
                    ablation, patching, confound), experiment framework
scripts/            Entry points: data prep and experiment drivers
config/experiments/ YAML config, one per experiment
extra_analyses/     Appendix K and L analyses, with their result JSONs
results/            Published summary JSONs (figures regenerate from these)
_paper/             LaTeX source, figures, and gen_figures.py
```

## Supplementary code

Two pieces ship that back no table in the paper, kept because they are honest parts of the analysis rather than because a claim depends on them:

- **`confound`** (`config/experiments/confound.yaml`, `raid_analysis/evaluation/confound.py`) — tests selected neurons against text-length and domain confounds. Runs, reports nothing in the paper.
- **`raid_analysis/evaluation/causal.py`** is shared machinery for ablation and patching, not a standalone experiment.

## Licence

MIT — see [LICENSE](LICENSE). RAID is separately MIT-licensed by its authors; no RAID text or model weights are redistributed in this repository.
