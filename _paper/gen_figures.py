"""Generate all paper figures as PDF (for LaTeX) and PNG (for the README).

Both formats are written to _paper/figures/ on every run, so the README
images never drift from the PDFs the paper embeds.
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent
EXP    = ROOT / 'results/experiments/20260510_rerun_main'
LOGO_E = ROOT / 'results/experiments/20260513_190517'
CHAR_J = EXP / 'characterize/characterize_results.json'
OUT    = Path(__file__).parent / 'figures'
OUT.mkdir(exist_ok=True)

GENERATORS_ALL = ['gpt4', 'gpt2', 'mpt', 'mistral-chat', 'llama-chat', 'cohere-chat']
# Only these have patching data
GENERATORS_PATCH = ['gpt4', 'gpt2', 'mpt', 'mistral-chat', 'llama-chat', 'cohere-chat']
PRETTY = {
    'gpt4': 'GPT-4', 'gpt2': 'GPT-2', 'mpt': 'MPT',
    'mistral-chat': 'Mistral', 'llama-chat': 'LLaMA', 'cohere-chat': 'Cohere'
}

# Instruction-tuned vs base
INST = {'gpt4', 'mistral-chat', 'llama-chat', 'cohere-chat'}
BASE = {'gpt2', 'mpt'}

# ── shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

COL_INST = '#2166ac'   # blue  – instruction-tuned
COL_BASE = '#d6604d'   # red   – base

def gen_color(g):
    return COL_INST if g in INST else COL_BASE

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Flip rate vs k  (log x-axis)
# ════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: flip rate vs k …")

# The last k point in each sweep equals that cell's stable-neuron count and
# varies across folds/seeds.  Use only the fixed prefix [1,2,5,10,20,50].
FIXED_K = [1, 2, 5, 10, 20, 50]

sel_means = {}
rnd_means  = {}

for g in GENERATORS_PATCH:
    sel_cells, rnd_cells = [], []
    for seed_dir in sorted(glob.glob(str(EXP / f'patching/{g}/seed_*'))):
        for fold_dir in sorted(glob.glob(seed_dir + '/fold_*')):
            mf = fold_dir + '/eval_metrics.json'
            if not os.path.exists(mf):
                continue
            with open(mf) as f:
                m = json.load(f)
            sweep = m['metrics']['patching_sweep']
            # index by k value for safety
            sweep_map = {s['k']: s for s in sweep}
            if not all(k in sweep_map for k in FIXED_K):
                continue
            sel_cells.append([sweep_map[k]['selected_flip_rate_mean'] * 100 for k in FIXED_K])
            rnd_cells.append([sweep_map[k]['random_flip_rate_mean']   * 100 for k in FIXED_K])

    sel_means[g] = np.mean(sel_cells, axis=0)
    rnd_means[g] = np.mean(rnd_cells, axis=0)
    print(f"  {PRETTY[g]}: {len(sel_cells)} cells, sel={[round(x,2) for x in sel_means[g]]}")

k_arr = np.array(FIXED_K)

# ── curve-shape diagnostic ──
print("")
print("  Curve shape - R2(log10(k), flip_rate). Descriptive only;")
print("  no claim in the paper depends on these values:")
log_k = np.log10(k_arr)
r2_vals = {}
for g in GENERATORS_PATCH:
    y = sel_means[g]
    r2 = np.corrcoef(log_k, y)[0, 1] ** 2
    r2_vals[g] = r2
    print(f"  {PRETTY[g]:8s}  R2={r2:.3f}")


# ── plot ──
linestyles = ['-', '--', '-.', ':', (0,(3,1,1,1)), (0,(5,1))]
markers    = ['o', 's', 'D', '^', 'v', 'P']

fig, ax = plt.subplots(figsize=(3.4, 2.6))
for i, g in enumerate(GENERATORS_PATCH):
    ax.plot(k_arr, sel_means[g],
            marker=markers[i], markersize=4,
            linestyle=linestyles[i % len(linestyles)],
            color=gen_color(g),
            label=PRETTY[g], linewidth=1.4)

# Per-line selected/random ratio at the rightmost k (k=50)
print("\n  selected/random ratio at k=50:")
ratio_info = []
for i, g in enumerate(GENERATORS_PATCH):
    ratio = sel_means[g][-1] / max(rnd_means[g][-1], 1e-6)
    ratio_info.append((i, g, ratio, sel_means[g][-1]))
    print(f"  {PRETTY[g]:8s}  {ratio:.1f}x")

# Place ratio labels in data coords with guaranteed non-overlap.
# Compute minimum y-separation in data units (8 pt gap, figure is 2.6in tall).
y_max_data = max(sel_means[g][-1] for g in GENERATORS_PATCH) * 1.15
min_sep_data = 8.0 / (2.6 * 72 / y_max_data)  # 8pt → data units

ratio_info_sorted = sorted(ratio_info, key=lambda t: t[3])
label_y_data = []
for j, (_, _, _, y_val) in enumerate(ratio_info_sorted):
    if j == 0:
        label_y_data.append(y_val)
    else:
        label_y_data.append(max(y_val, label_y_data[-1] + min_sep_data))

label_y_map = {info[1]: label_y_data[k] for k, info in enumerate(ratio_info_sorted)}

for i, g, ratio, y_val in ratio_info:
    ax.annotate(f'{ratio:.0f}x',
                xy=(k_arr[-1], y_val),
                xytext=(k_arr[-1] * 1.12, label_y_map[g]),
                textcoords='data',
                fontsize=7, color=gen_color(g),
                va='center', ha='left')

ax.set_xscale('log')
ax.set_xticks(FIXED_K)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel('Number of patched neurons k')
ax.set_ylabel('Flip rate (%)')
ax.legend(loc='upper left', frameon=False, fontsize=7.5)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(0.8, 130)

plt.tight_layout(pad=0.4)
fig.savefig(OUT / 'fig_flip_rate.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_flip_rate.png', bbox_inches='tight', dpi=200)
plt.close()
print(f"\n  Saved -> {OUT / 'fig_flip_rate.pdf'}")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Layer distribution stacked horizontal bar
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 3: layer distribution …")

with open(CHAR_J) as f:
    char = json.load(f)

layer_dist = char['layer_distributions']   # {gen: {layer_str: count}}

# Order: instruction-tuned then base (same as paper)
gen_order = [g for g in GENERATORS_ALL if g in layer_dist]
# Re-order: inst first
gen_order = (
    [g for g in gen_order if g in INST] +
    [g for g in gen_order if g in BASE]
)

row_early, row_mid, row_late = [], [], []
for g in gen_order:
    ld = layer_dist[g]
    early = sum(v for k, v in ld.items() if int(k) <= 10)
    mid   = ld.get('11', 0)
    late  = ld.get('12', 0)
    tot   = early + mid + late
    row_early.append(early / tot * 100)
    row_mid.append(mid   / tot * 100)
    row_late.append(late  / tot * 100)

row_early = np.array(row_early)
row_mid   = np.array(row_mid)
row_late  = np.array(row_late)

pretty_labels = []
for g in gen_order:
    tag = '(IT)' if g in INST else '(base)'
    pretty_labels.append(f'{PRETTY[g]} {tag}')

n = len(gen_order)
y_pos = np.arange(n)

c_early = '#c6dbef'
c_mid   = '#6baed6'
c_late  = '#08519c'

fig, ax = plt.subplots(figsize=(3.5, 3.3))
ax.barh(y_pos, row_early, color=c_early, label='Layers 1–10', edgecolor='white', linewidth=0.3)
ax.barh(y_pos, row_mid,  left=row_early,              color=c_mid, label='Layer 11', edgecolor='white', linewidth=0.3)
ax.barh(y_pos, row_late, left=row_early + row_mid,    color=c_late, label='Layer 12', edgecolor='white', linewidth=0.3)

# Percentage labels on the L12 segment
for i in range(n):
    pct = row_late[i]
    if pct > 3:
        left = row_early[i] + row_mid[i]
        ax.text(left + pct / 2, i, f'{pct:.0f}%',
                ha='center', va='center', fontsize=7, color='white', fontweight='bold')

# Separator between instruction-tuned and base groups
n_inst = sum(1 for g in gen_order if g in INST)
ax.axhline(n_inst - 0.5, color='#555555', linewidth=0.8, linestyle='--')
ax.text(101, n_inst - 0.5 + 0.05, 'IT / base', va='bottom', ha='left', fontsize=7, color='#555555')

ax.set_yticks(y_pos)
ax.set_yticklabels(pretty_labels, fontsize=8)
ax.set_xlabel('% of stable neurons')
ax.set_xlim(0, 115)
ax.spines[['top', 'right']].set_visible(False)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper center',
          bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=7.5,
          columnspacing=1.2)

plt.tight_layout(pad=0.4)
fig.savefig(OUT / 'fig_layer_dist.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_layer_dist.png', bbox_inches='tight', dpi=200)
plt.close()
print(f"  Saved -> {OUT / 'fig_layer_dist.pdf'}")

# Print layer data for verification
for i, g in enumerate(gen_order):
    print(f"  {PRETTY[g]:8s}  early={row_early[i]:.1f}%  L11={row_mid[i]:.1f}%  L12={row_late[i]:.1f}%")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Jaccard heatmap
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 5: Jaccard heatmap …")

jac     = char['jaccard']
jac_gens = jac['generators']
jac_mat  = np.array(jac['matrix'])

# Reorder: instruction-tuned first
order = (
    [g for g in jac_gens if g in INST] +
    [g for g in jac_gens if g in BASE]
)
idx   = [jac_gens.index(g) for g in order]
mat_r = jac_mat[np.ix_(idx, idx)]
labels = [PRETTY[g] for g in order]
n_inst = sum(1 for g in order if g in INST)
n_base = sum(1 for g in order if g in BASE)

print("  Jaccard matrix (reordered):")
for i, g in enumerate(order):
    row_str = '  '.join(f'{mat_r[i,j]:.3f}' for j in range(len(order)))
    print(f"  {PRETTY[g]:8s}  {row_str}")

fig, ax = plt.subplots(figsize=(3.3, 2.9))

# Off-diagonal max for vmax
off_diag = mat_r[~np.eye(len(order), dtype=bool)]
vmax = min(off_diag.max() * 1.15, 0.5)

cmap = plt.cm.Blues
im = ax.imshow(mat_r, cmap=cmap, vmin=0, vmax=vmax)

ax.set_xticks(range(len(order)))
ax.set_yticks(range(len(order)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(labels, fontsize=8)

# Annotate
for i in range(len(order)):
    for j in range(len(order)):
        val = mat_r[i, j]
        if i == j:
            ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='#aaaaaa')
        else:
            text_color = 'white' if val > vmax * 0.65 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=6.5, color=text_color)

# Draw dashed rectangles around the IT and base diagonal blocks
rect_inst = plt.Rectangle((-0.5, -0.5), n_inst, n_inst,
                           fill=False, edgecolor=COL_INST, linewidth=1.6, linestyle='--')
rect_base = plt.Rectangle((n_inst - 0.5, n_inst - 0.5), n_base, n_base,
                           fill=False, edgecolor=COL_BASE, linewidth=1.6, linestyle='--')
ax.add_patch(rect_inst)
ax.add_patch(rect_base)

cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
cb.set_label('Jaccard', fontsize=8)
cb.ax.tick_params(labelsize=7)
ax.spines[:].set_visible(False)

plt.tight_layout(pad=0.4)
fig.savefig(OUT / 'fig_jaccard.pdf', bbox_inches='tight')
fig.savefig(OUT / 'fig_jaccard.png', bbox_inches='tight', dpi=200)
plt.close()
print(f"  Saved -> {OUT / 'fig_jaccard.pdf'}")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — LOGO generalisation bar chart
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 4: LOGO …")

logo_path = LOGO_E / 'logo/summary.json'
if not logo_path.exists():
    # Try alternate location
    candidates = list(LOGO_E.rglob('summary.json'))
    print(f"  summary.json not at expected path; found: {candidates[:5]}")
    logo_path = candidates[0] if candidates else None

if logo_path:
    with open(logo_path) as f:
        logo = json.load(f)
    print(f"  Loaded from {logo_path}")
    print(f"  Top-level keys: {list(logo.keys())[:10]}")

    # Inspect structure
    first_key = list(logo.keys())[0]
    print(f"  First entry ({first_key!r}):", json.dumps(logo[first_key], indent=2)[:500])
else:
    print("  Could not find LOGO summary.json — skipping Figure 4")
    logo = None

if logo is not None:
    # Results is a list of dicts with keys: held_out, full_l2_accuracy_mean, selected_l2_accuracy_mean
    results = logo.get('results', [])

    # Pretty family names
    FAM_PRETTY = {
        'family_cohere': 'Cohere',
        'family_gpt':    'GPT',
        'family_llama':  'LLaMA',
        'family_mistral': 'Mistral',
        'family_mpt':    'MPT',
    }

    full_accs, sel_accs, fam_labels = [], [], []
    for entry in results:
        fam = entry.get('held_out', '')
        fa  = entry.get('full_l2_accuracy_mean')
        sa  = entry.get('selected_l2_accuracy_mean')
        if fa is not None and sa is not None:
            full_accs.append(fa * 100)
            sel_accs.append(sa * 100)
            fam_labels.append(FAM_PRETTY.get(fam, fam.replace('family_', '').capitalize()))

    print(f"  Parsed {len(full_accs)} families:")
    for fam, fa, sa in zip(fam_labels, full_accs, sel_accs):
        pct_ret = sa / fa * 100
        print(f"    {fam:10s}  full={fa:.1f}%  sel={sa:.1f}%  retention={pct_ret:.1f}%")

    if full_accs:
        n = len(fam_labels)
        x = np.arange(n)
        w = 0.35

        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        bars_f = ax.bar(x - w/2, full_accs, width=w, label='Full L2',
                        color='#4dac26', alpha=0.85, edgecolor='white')
        bars_s = ax.bar(x + w/2, sel_accs,  width=w, label='Sparse L2',
                        color='#b8e186', alpha=0.85, edgecolor='white')

        # Accuracy on top of both bars; retention printed under the sparse value
        for xi, fa, sa in zip(x, full_accs, sel_accs):
            ax.text(xi - w/2, fa + 0.5, f'{fa:.1f}', ha='center', va='bottom',
                    fontsize=6.5, color='#333333')
            ax.text(xi + w/2, sa + 0.5, f'{sa:.1f}', ha='center', va='bottom',
                    fontsize=6.5, color='#333333')
            ax.text(xi + w/2, sa - 1.2, f'{sa / fa * 100:.0f}%', ha='center',
                    va='top', fontsize=6, color='#4d4d4d')

        ax.set_xticks(x)
        ax.set_xticklabels(fam_labels, rotation=0, ha='center', fontsize=8)
        ax.set_ylabel('Accuracy (%)')
        ylo = max(0, min(full_accs + sel_accs) - 6)
        ax.set_ylim(bottom=ylo, top=max(full_accs) + 3.5)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
        # legend above the axes so it cannot sit on top of the bars
        ax.legend(frameon=False, fontsize=8, ncol=2, loc='lower center',
                  bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0,
                  handlelength=1.2, columnspacing=1.4)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', length=0)

        plt.tight_layout(pad=0.3)
        fig.savefig(OUT / 'fig_logo.pdf', bbox_inches='tight')
        fig.savefig(OUT / 'fig_logo.png', bbox_inches='tight', dpi=200)
        plt.close()
        print(f"  Saved -> {OUT / 'fig_logo.pdf'}")

print("\nAll figures done.")
