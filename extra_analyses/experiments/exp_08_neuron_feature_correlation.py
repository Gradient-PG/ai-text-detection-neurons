"""What surface features do the selected neurons track?

Correlates the highest-weight stable neurons against simple surface statistics
of the input text, to test how much of their behaviour a single surface cue
accounts for. Reported in the paper as Appendix K.

Protocol
--------
For each generator:
  1. Load the stable set S* from the canonical sparse-probe run.
  2. Rank stable neurons by mean |L1 selection-probe weight| over the 15
     (seed, fold) cells; keep the top 10.
  3. Compute four surface statistics per sample: type-token ratio, mean word
     length, mean sentence length, punctuation density.
  4. Spearman-correlate each of the 10 neurons against each of the 4 features
     (40 tests per generator), then apply Benjamini-Hochberg FDR at q = 0.05
     within the generator.

Texts are reconstructed by decoding the cached ``input_ids`` (the exact token
sequence BERT encoded, truncated at max_len), so features are index-aligned
with the activation matrix by construction. The reconstruction is lowercased
and wordpiece-detokenized; that is a limitation for casing-sensitive features
but none of the four used here depends on case.

Run from the repository root:
    python extra_analyses/experiments/exp_08_neuron_feature_correlation.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from scipy.stats import spearmanr
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "results" / "experiments" / "20260510_rerun_main" / "sparse_probe"
ACTS = REPO / "results"
PROCESSED = REPO / "data" / "processed"
OUT = REPO / "extra_analyses" / "results" / "exp_08_neuron_feature_correlation.json"

GENERATORS = {
    "gpt4": "raid_gpt4",
    "gpt2": "raid_gpt2",
    "mpt": "raid_mpt",
    "mistral-chat": "raid_mistral_chat",
    "llama-chat": "raid_llama_chat",
    "cohere-chat": "raid_cohere_chat",
}
PRETTY = {
    "gpt4": "GPT-4", "gpt2": "GPT-2", "mpt": "MPT",
    "mistral-chat": "Mistral", "llama-chat": "LLaMA", "cohere-chat": "Cohere",
}
FEATURES = ["type_token_ratio", "mean_word_length", "mean_sentence_length",
            "punctuation_density"]
N_TOP = 10
Q = 0.05
PUNCT = set(".,;:!?'\"()[]{}-–—/\\&%$#@*+=<>|~`")

_SENT_SPLIT = re.compile(r"[.!?]+")


def text_features(text: str) -> dict[str, float]:
    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return {f: np.nan for f in FEATURES}
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    n_chars = len(text)
    return {
        "type_token_ratio": len({w.lower() for w in words}) / n_words,
        "mean_word_length": float(np.mean([len(w) for w in words])),
        "mean_sentence_length": n_words / max(len(sents), 1),
        "punctuation_density": sum(c in PUNCT for c in text) / max(n_chars, 1),
    }


def bh_fdr(pvals: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg: return a boolean mask of rejected nulls."""
    n = len(pvals)
    order = np.argsort(pvals)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = pvals[order] <= thresh
    reject = np.zeros(n, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        reject[order[: kmax + 1]] = True
    return reject


def load_stable_and_weights(gen: str) -> list[tuple[int, int, float]]:
    """Stable neurons ranked by mean |L1 weight| over the 15 cells."""
    agg = json.loads((RUN / gen / "aggregate.json").read_text())
    stable = [tuple(x) for x in agg["stable_neurons"]]

    sums: dict[tuple[int, int], list[float]] = {n: [] for n in stable}
    for seed_dir in sorted((RUN / gen).glob("seed_*")):
        for fold_dir in sorted(seed_dir.glob("fold_*")):
            w = np.load(fold_dir / "probe_weights.npy").ravel()
            sel = json.loads((fold_dir / "selection.json").read_text())
            idx = sel.get("neuron_indices_global", [])
            # probe_weights.npy is the full 9,216-dim L1 coefficient vector when
            # its length matches; otherwise it is aligned with the selected set.
            for layer, unit in stable:
                flat = (layer - 1) * 768 + unit
                if w.size == 9216:
                    sums[(layer, unit)].append(abs(float(w[flat])))
                elif flat in idx:
                    sums[(layer, unit)].append(abs(float(w[idx.index(flat)])))
    ranked = sorted(
        ((l, u, float(np.mean(v)) if v else 0.0) for (l, u), v in sums.items()),
        key=lambda t: -t[2],
    )
    return ranked[:N_TOP]


def replay_extraction_order(dataset, n_samples: int):
    """Reproduce the row order of the cached activation matrices.

    ``ActivationExtractor`` shuffles and subsamples before encoding, so the
    tokenized dataset on disk is *not* in activation order. This replays the
    same seed-42 draw used by the extractor (subsample each class, then shuffle
    the concatenation), which is verified against the cached labels by the
    caller. If this replay ever drifts from the extractor, the label check
    fails loudly rather than silently misattributing neurons to texts.
    """
    from datasets import concatenate_datasets

    ai = dataset.filter(lambda x: x["label"] == 1)
    human = dataset.filter(lambda x: x["label"] == 0)
    per_class = n_samples // 2
    ai_subset = ai.shuffle(seed=42).select(range(min(per_class, len(ai))))
    human_subset = human.shuffle(seed=42).select(range(min(per_class, len(human))))
    return concatenate_datasets([ai_subset, human_subset]).shuffle(seed=42)


def load_activations(slug: str, layers_units: list[tuple[int, int]]) -> np.ndarray:
    """(N, len(layers_units)) matrix of the requested CLS dimensions."""
    root = ACTS / f"activations_{slug}"
    cols = []
    cache: dict[int, np.ndarray] = {}
    for layer, unit in layers_units:
        if layer not in cache:
            cache[layer] = np.load(root / f"layer_{layer}_activations.npy", mmap_mode="r")
        cols.append(np.asarray(cache[layer][:, unit]))
    return np.column_stack(cols)


def main() -> None:
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    results = {}

    for gen, slug in GENERATORS.items():
        print(f"=== {gen} ===")
        top = load_stable_and_weights(gen)
        units = [(l, u) for l, u, _ in top]

        labels = np.load(ACTS / f"activations_{slug}" / "labels.npy")
        ds = load_from_disk(str(PROCESSED / slug))["train"].with_format(None)
        ds = replay_extraction_order(ds, len(labels))
        assert (np.asarray(ds["label"]) == labels).all(), \
            f"{gen}: replayed order does not match cached activation order"
        texts = tok.batch_decode(list(ds["input_ids"]), skip_special_tokens=True)
        feats = np.array([[text_features(t)[f] for f in FEATURES] for t in texts])

        acts = load_activations(slug, units)
        assert len(acts) == len(feats), (len(acts), len(feats))
        print(f"  {len(acts)} samples, {len(units)} neurons")

        rhos = np.zeros((len(units), len(FEATURES)))
        pvals = np.zeros_like(rhos)
        for i in range(len(units)):
            for j in range(len(FEATURES)):
                ok = np.isfinite(feats[:, j])
                r, p = spearmanr(acts[ok, i], feats[ok, j])
                rhos[i, j], pvals[i, j] = r, p

        reject = bh_fdr(pvals.ravel(), Q).reshape(pvals.shape)
        surviving = np.where(reject, np.abs(rhos), 0.0)
        bi, bj = np.unravel_index(np.argmax(surviving), surviving.shape)

        # how strongly each feature separates the classes on its own — context
        # for how much of a neuron-feature association is even available
        feature_label_rho = {
            FEATURES[j]: float(spearmanr(feats[np.isfinite(feats[:, j]), j],
                                         labels[np.isfinite(feats[:, j])]).statistic)
            for j in range(len(FEATURES))
        }
        per_feature = {
            FEATURES[j]: float(surviving[:, j].max()) for j in range(len(FEATURES))
        }
        results[gen] = {
            "n_samples": int(len(acts)),
            "top_neurons_layer_unit_weight": [[l, u, w] for l, u, w in top],
            "n_tests": int(rhos.size),
            "n_surviving_bh": int(reject.sum()),
            "any_surviving": bool(reject.any()),
            "strongest": {
                "neuron": [int(units[bi][0]), int(units[bi][1])],
                "feature": FEATURES[bj],
                "rho": float(rhos[bi, bj]),
                "p": float(pvals[bi, bj]),
                "r2_approx": float(rhos[bi, bj] ** 2),
                "survives_bh": bool(reject[bi, bj]),
            },
            "per_feature_max_abs_surviving_rho": per_feature,
            "top_feature": max(per_feature, key=per_feature.get),
            "feature_vs_label_rho": feature_label_rho,
            "rho_matrix": rhos.tolist(),
            "p_matrix": pvals.tolist(),
        }
        s = results[gen]["strongest"]
        print(f"  strongest: L{s['neuron'][0]}u{s['neuron'][1]} x {s['feature']} "
              f"rho={s['rho']:+.3f} (r2={s['r2_approx']:.3f}), "
              f"{results[gen]['n_surviving_bh']}/{results[gen]['n_tests']} survive BH")
        print("  feature vs label: " + ", ".join(
            f"{k}={v:+.3f}" for k, v in feature_label_rho.items()))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "protocol": {
            "n_top_neurons_by_weight": N_TOP,
            "features": FEATURES,
            "correlation": "spearman",
            "multiple_comparison": f"benjamini-hochberg q={Q}, within generator",
            "text_source": "decoded cached input_ids (bert-base-uncased)",
            "source_run": "results/experiments/20260510_rerun_main",
        },
        "results": results,
    }, indent=2))
    print(f"\nSaved -> {OUT}")

    print("\n--- summary ---")
    for gen in GENERATORS:
        r = results[gen]
        print(f"{PRETTY[gen]:8s} top_feature={r['top_feature']:22s} "
              f"|rho|max={abs(r['strongest']['rho']):.3f} "
              f"rho={r['strongest']['rho']:+.3f}")


if __name__ == "__main__":
    main()
