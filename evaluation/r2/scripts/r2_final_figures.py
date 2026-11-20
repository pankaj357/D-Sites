"""R2 final figure regeneration (EVB-25-0116.R2 correction pass).
Regenerates Figures 2-5 from the deposited R2 statistics ONLY (no new analyses).
Outputs: figures/r2/final/figure{N}.png (300 dpi) + .pdf
"""
import json, csv, ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'axes.linewidth': 0.9, 'lines.linewidth': 1.6, 'figure.dpi': 300})
R = 'results/r2'
EV = '../../evaluation/results'
OUT = 'figures/r2/final'
import os; os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)

# ---------------- Figure 2 ----------------
pr = json.load(open(f'{R}/paired_site_recovery.json'))
cv = {r['TF']: r for r in csv.DictReader(open(f'{EV}/cv_summary.csv'))}
order = ['AmrZ', 'GlxR', 'CodY', 'FNR_ecoli']; labels = ['AmrZ', 'GlxR', 'CodY', 'FNR']
rec = [float(cv[t]['recall_in_space_mean']) for t in order]
recsd = [float(cv[t]['recall_in_space_sd']) for t in order]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
ax = axes[0]; x = np.arange(4)
b = ax.bar(x, rec, 0.55, color='#4472C4', yerr=recsd, capsize=4, error_kw=dict(lw=1.1))
for i, bar in enumerate(b):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+recsd[i]+0.02, f"{rec[i]:.2f}",
            ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.12)
ax.set_ylabel('Recall (in promoter space)\nheld-out, 5-fold grouped CV')
ax.set_title('A', loc='left', fontweight='bold')
ax = axes[1]; w = 0.38
d_sites = [pr['factors'][t]['dsites']['recovered_sites'] for t in order]
f_sites = [pr['factors'][t]['fimo']['recovered_sites'] for t in order]
Ns = [pr['factors'][t]['N'] for t in order]
ptxt = []
for t in order:
    hp = pr['factors'][t]['paired']['mcnemar_p_holm']
    ptxt.append(f"p={hp:.3g}" if hp>=0.001 else f"p={hp:.1e}")
x = np.arange(4)
ax.bar(x-w/2, d_sites, w, label='D-Sites top-N', color='#4472C4')
ax.bar(x+w/2, f_sites, w, label='FIMO top-N', color='#C00000')
for i in range(4):
    ax.text(i, max(d_sites[i], f_sites[i])+2.5, ptxt[i], ha='center', fontsize=7.2, color='#333333')
    ax.text(i, -10.5, f'N={Ns[i]}', ha='center', fontsize=7.2, color='#333333', clip_on=False)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, max(max(d_sites), max(f_sites))*1.25)
ax.set_ylabel('Known sites recovered at N')
ax.set_title('B', loc='left', fontweight='bold')
ax.legend(frameon=False, fontsize=7.5, loc='upper left')
fig.tight_layout()
fig.subplots_adjust(bottom=0.22)
save(fig, 'figure2')

# ---------------- Figure 3 (ablation; 3 CollecTF factors) ----------------
TFS3 = ['AmrZ', 'GlxR', 'CodY']
sets = ['PWM_only', 'PWM_GC_Entropy', 'all_11']
setlabels = ['PWM only', 'PWM+GC+entropy', 'All 11 features']
colors = ['#8FBC8F', '#F4C28A', '#4472C4']
ap = {s: [] for s in sets}; apsd = {s: [] for s in sets}
tp = {s: [] for s in sets}; tpsd = {s: [] for s in sets}
for tf in TFS3:
    d = json.load(open(f'{EV}/ablation_{tf}.json'))
    fs = d['feature_sets']
    for s in sets:
        ap[s].append(fs[s]['summary']['PR_AUC']['mean']); apsd[s].append(fs[s]['summary']['PR_AUC']['sd'])
        tp[s].append(fs[s]['summary']['top5pct_recall']['mean']); tpsd[s].append(fs[s]['summary']['top5pct_recall']['sd'])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, metric, sds, ylab, ttl in ((axes[0], ap, apsd, 'Average precision (5-fold CV, mean ± SD)', 'A. Average precision'),
                                   (axes[1], tp, tpsd, 'Recall of top 5% predictions (mean ± SD)', 'B. Top-5% recall')):
    x = np.arange(3); w = 0.26
    for j, s in enumerate(sets):
        ax.bar(x+(j-1)*w, metric[s], w, yerr=sds[s], capsize=3, color=colors[j],
               label=setlabels[j], error_kw=dict(lw=1.0))
    ax.set_xticks(x); ax.set_xticklabels(TFS3); ax.set_ylabel(ylab)
    ax.set_title(ttl, loc='left', fontweight='bold')
axes[0].legend(frameon=False, fontsize=7.5)
fig.suptitle('Feature ablation', fontsize=11, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.94])
save(fig, 'figure3')

# ---------------- Figure 4 (site-level PR, primary) ----------------
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
panels = ['A. AmrZ', 'B. GlxR', 'C. CodY', 'D. FNR']
for ax, tf, ttl in zip(axes.flat, order, panels):
    d = json.load(open(f'{R}/sitelevel_pr_{tf}.json'))
    a = d['variant_a_floor_at_threshold']
    cp = a['curve_points']
    ax.plot(cp['dsites_recall'], cp['dsites_precision'], color='#4472C4',
            label=f"D-Sites (AP={a['dsites']['site_AP']:.3f})")
    ax.plot(cp['fimo_recall'], cp['fimo_precision'], color='#C00000',
            label=f"FIMO (AP={a['fimo']['site_AP']:.3f})")
    ns = a['dsites']['no_skill']
    ax.axhline(ns, color='gray', ls='--', lw=1.0, label=f'No-skill ({ns:.3f})')
    ax.set_xlim(0, 1.01)
    lo = min(ns*0.97, min(min(cp['dsites_precision']), min(cp['fimo_precision']))*0.99)
    ax.set_ylim(lo, 1.012)
    ax.set_xlabel('Site-level recall'); ax.set_ylabel('Site-level precision')
    ax.set_title(ttl, loc='left', fontweight='bold')
    ax.grid(alpha=0.3, lw=0.5)
handles, lbls = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc='lower center', ncol=3, frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.045, 1, 1])
save(fig, 'figure4')

# ---------------- Figure 5 (FNR operon-level validation) ----------------
def as_dict(v):
    return v if isinstance(v, dict) else ast.literal_eval(v)

m2 = json.load(open(f'{R}/fnr_operon_matched819.json'))
fisher = as_dict(m2['fisher_per_method'])
bg_rates = as_dict(m2['background_rate'])
curve = json.load(open(f'{R}/fnr_enrichment_curve_fullset_fraction.json'))['curve']
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
ax = axes[0]
methods = [('fimo', 'FIMO p<1e-4', '#C00000'), ('hybrid', 'D-Sites hybrid', '#4472C4'),
           ('pwm_only', 'D-Sites no-shape', '#70AD47')]
x = np.arange(3); w = 0.26
groups = [('background', 'Background (2,750)', '#A6A6A6'),
          ('DEP_all', 'All regulated (109)', '#4472C4'),
          ('DEP_activated', 'Activated (49)', '#70AD47')]
for j, (g, lab, col) in enumerate(groups):
    vals = []
    for key, _, _ in methods:
        if g == 'background':
            vals.append(bg_rates[key])
        else:
            vals.append(fisher[key][g]['group_rate'])
    ax.bar(x+(j-1)*w, vals, w, label=lab, color=col)
ax.set_xticks(x); ax.set_xticklabels([m[1] for m in methods], fontsize=7.5)
ax.set_ylabel('Operon hit rate'); ax.set_ylim(0, 1.1)
ax.set_title('A. Matched-count comparison', loc='left', fontweight='bold')
ax.legend(frameon=False, fontsize=7, loc='upper left')
ax = axes[1]
frac = [c['fraction_of_full_set'] for c in curve]
for key, lab, col, mk in [('DEP_all_fold', 'All regulated (109)', '#4472C4', 'o'),
                          ('DEP_activated_fold', 'Activated (49)', '#70AD47', 's'),
                          ('DEP_repressed_fold', 'Repressed (60)', '#ED7D31', '^')]:
    ax.plot(frac, [c[key] for c in curve], marker=mk, color=col, label=lab, ms=4)
ax.axhline(1, color='gray', ls=':', lw=1)
ax.set_xscale('log'); ax.set_xticks(frac)
ax.set_xticklabels([f'{f*100:g}%' for f in frac], fontsize=7)
ax.set_xlabel('Fraction of full prediction set retained (39,482)')
ax.set_ylabel('Fold enrichment vs background')
ax.set_title('B. Enrichment across prediction subsets', loc='left', fontweight='bold')
ax.legend(frameon=False, fontsize=7.5, loc='upper right')
fig.suptitle('FNR operon-level validation', fontsize=11, fontweight='bold')
fig.tight_layout(rect=[0, 0.06, 1, 0.94])
save(fig, 'figure5')
print('ALL FINAL FIGURES DONE')
