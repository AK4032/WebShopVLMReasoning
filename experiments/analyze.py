"""Tables and figures for the reasoning-effort / calibration / concentration questions.

Reads results/scored/<split>.csv and writes CSV tables and PNG figures to
results/analysis/<split>/. Success = WebShop reward of exactly 1.0. Confidence
is the model's 0-10 rating, rescaled to 0-1 for calibration metrics.

Usage (repo root; pandas, matplotlib, scipy):
    python experiments/analyze.py --split dev
"""
import argparse
import itertools
import os
from os.path import abspath, dirname, join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

REPO_DIR = dirname(dirname(abspath(__file__)))
EFFORTS = ['low', 'medium', 'high']
N_BOOT = 2000


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def ece(conf, correct, n_bins=10):
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    return sum(abs(conf[bins == b].mean() - correct[bins == b].mean()) * (bins == b).mean()
               for b in range(n_bins) if (bins == b).any())


def auroc(conf, correct):
    pos, neg = conf[correct], conf[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return mannwhitneyu(pos, neg, alternative='two-sided').statistic / (len(pos) * len(neg))


def calibration(conf, correct, suffix=''):
    if len(conf) == 0:
        return {f'{k}{suffix}': np.nan for k in ['success', 'overconfidence', 'ece', 'brier', 'auroc']}
    return {f'success{suffix}': correct.mean(), f'overconfidence{suffix}': conf.mean() - correct.mean(),
            f'ece{suffix}': ece(conf, correct), f'brier{suffix}': np.mean((conf - correct) ** 2),
            f'auroc{suffix}': auroc(conf, correct)}


def summarize(df):
    rows = []
    for (model, effort), g in df.groupby(['model_key', 'effort']):
        v = g[g['valid']]
        conf = (v['confidence'] / 10).clip(0, 1).to_numpy(float)
        correct = v['success'].to_numpy(bool)
        lo, hi = wilson(correct.sum(), len(v))
        row = {
            'model_key': model, 'effort': effort, 'n': len(g), 'valid_rate': g['valid'].mean(),
            'success_ci_low': lo, 'success_ci_high': hi,
            'success_invalid_as_fail': g['success'].fillna(False).astype(bool).mean(),
            'mean_reward': v['reward'].mean(), 'chose_target': v['chose_target'].astype(bool).mean(),
            'chose_perfect_product': v['chosen_product_perfect'].astype(bool).mean(),
            'mean_confidence': v['confidence'].mean(),
            'mean_reasoning_tokens': g['reasoning_tokens'].mean(),
            'mean_completion_tokens': g['completion_tokens'].mean(),
        }
        row.update(calibration(conf, correct))
        if 'success_lenient' in v:
            row.update(calibration(conf, v['success_lenient'].to_numpy(bool), '_lenient'))
            row['mean_reward_lenient'] = v['reward_lenient'].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    out['effort'] = pd.Categorical(out['effort'], EFFORTS, ordered=True)
    return out.sort_values(['model_key', 'effort'])


def effort_effects(df, rng):
    """Paired (same tasks) high-minus-low differences with bootstrap 95% CIs."""
    rows = []
    for model, g in df[df['valid']].groupby('model_key'):
        metrics = [m for m in ['success', 'success_lenient', 'confidence'] if m in g]
        wide = g.pivot_table(index=['task_id', 'sample'], columns='effort',
                             values=metrics, aggfunc='first')
        for a, b in [('low', 'high'), ('low', 'medium'), ('medium', 'high')]:
            if a not in wide['success'] or b not in wide['success']:
                continue
            for metric in metrics:
                d = (wide[metric][b].astype(float) - wide[metric][a].astype(float)).dropna().to_numpy()
                if len(d) == 0:
                    continue
                boots = rng.choice(d, (N_BOOT, len(d))).mean(axis=1)
                rows.append({'model_key': model, 'contrast': f'{b} - {a}', 'metric': metric, 'n_tasks': len(d),
                             'diff': d.mean(), 'ci_low': np.percentile(boots, 2.5),
                             'ci_high': np.percentile(boots, 97.5)})
    return pd.DataFrame(rows)


def concentration(df):
    """Per-task choice concentration across all (model, effort) conditions, plus pairwise agreement."""
    v = df[df['valid'] & (df['sample'] == 0)].copy()
    v['condition'] = v['model_key'] + ' / ' + v['effort']
    choices = v.pivot_table(index='task_id', columns='condition', values='chosen_asin', aggfunc='first')
    per_task = []
    for task_id, row in choices.iterrows():
        shares = row.dropna().value_counts(normalize=True)
        per_task.append({'task_id': task_id, 'n_conditions': row.notna().sum(),
                         'n_distinct_products': len(shares), 'top_share': shares.iloc[0],
                         'hhi': (shares ** 2).sum()})
    conds = sorted(choices.columns, key=lambda c: (c.split(' / ')[0], EFFORTS.index(c.split(' / ')[1])))
    agree = pd.DataFrame(np.nan, index=conds, columns=conds)
    for a, b in itertools.product(conds, conds):
        if a == b:
            agree.loc[a, b] = 1.0
            continue
        both = choices[[a, b]].dropna()
        if len(both):
            agree.loc[a, b] = (both[a] == both[b]).mean()
    return pd.DataFrame(per_task), agree


def plot_effort(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for model, g in summary.groupby('model_key'):
        x = [EFFORTS.index(e) for e in g['effort']]
        bars = axes[0].errorbar(x, g['success'], yerr=[g['success'] - g['success_ci_low'], g['success_ci_high'] - g['success']],
                                marker='o', capsize=3, label=model)
        if 'success_lenient' in g:
            axes[0].plot(x, g['success_lenient'], linestyle='--', marker='x', color=bars[0].get_color())
        axes[1].plot(x, g['mean_confidence'], marker='o', label=model)
        axes[2].plot(x, g['mean_reasoning_tokens'], marker='o', label=model)
    titles = ['Success rate (solid: WebShop reward = 1.0; dashed: lenient)', 'Mean confidence (0-10)',
              'Mean reasoning tokens']
    for ax, title in zip(axes, titles):
        ax.set_xticks(range(3))
        ax.set_xticklabels(EFFORTS)
        ax.set_xlabel('reasoning effort')
        ax.set_title(title)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 10)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(df, path, n_bins=5):
    models = sorted(df['model_key'].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4.2), squeeze=False)
    edges = np.linspace(0, 1, n_bins + 1)
    for ax, model in zip(axes[0], models):
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1)
        for effort in EFFORTS:
            v = df[(df['model_key'] == model) & (df['effort'] == effort) & df['valid']]
            if v.empty:
                continue
            conf = (v['confidence'] / 10).clip(0, 1)
            bins = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
            pts = v.assign(b=bins, c=conf).groupby('b').agg(c=('c', 'mean'), s=('success', 'mean'), n=('c', 'size'))
            ax.plot(pts['c'], pts['s'], marker='o', label=effort)
        ax.set_title(model)
        ax.set_xlabel('stated confidence / 10')
        ax.set_ylabel('success rate')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_agreement(agree, path):
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(agree), 0.8 + 0.6 * len(agree)))
    im = ax.imshow(agree.to_numpy(float), vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(agree)))
    ax.set_xticklabels(agree.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(agree)))
    ax.set_yticklabels(agree.index, fontsize=8)
    ax.set_title('Share of tasks where two conditions chose the same product')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True)
    parser.add_argument('--scored', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    scored = args.scored or join(REPO_DIR, 'results', 'scored', f'{args.split}.csv')
    out_dir = args.out_dir or join(REPO_DIR, 'results', 'analysis', args.split)
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(scored)
    df['valid'] = df['valid'].astype(bool)
    rng = np.random.default_rng(args.seed)

    summary = summarize(df)
    summary.to_csv(join(out_dir, 'summary.csv'), index=False)
    by_difficulty = summarize(df[df['num_perfect_candidates'] == 1])
    by_difficulty.to_csv(join(out_dir, 'summary_single_correct_product.csv'), index=False)
    effects = effort_effects(df, rng)
    effects.to_csv(join(out_dir, 'effort_effects.csv'), index=False)
    per_task, agree = concentration(df)
    per_task.to_csv(join(out_dir, 'concentration_per_task.csv'), index=False)
    agree.to_csv(join(out_dir, 'pairwise_agreement.csv'))
    positions = (df[df['valid']].groupby(['model_key', 'effort'])['chosen_position']
                 .value_counts(normalize=True).unstack(fill_value=0))
    positions.to_csv(join(out_dir, 'chosen_position_distribution.csv'))

    plot_effort(summary, join(out_dir, 'effort_success_confidence_tokens.png'))
    plot_reliability(df, join(out_dir, 'reliability.png'))
    plot_agreement(agree, join(out_dir, 'agreement_heatmap.png'))

    cols = [c for c in ['model_key', 'effort', 'n', 'valid_rate', 'success', 'success_lenient',
                        'mean_confidence', 'overconfidence', 'overconfidence_lenient', 'ece',
                        'auroc', 'auroc_lenient', 'mean_reasoning_tokens'] if c in summary]
    print(summary[cols].round(3).to_string(index=False))
    if len(effects):
        print('\nEffort effects (paired, bootstrap 95% CI):')
        print(effects.round(3).to_string(index=False))
    if len(per_task):
        print(f'\nConcentration: mean top-product share {per_task["top_share"].mean():.2f}, '
              f'mean HHI {per_task["hhi"].mean():.2f} across {len(per_task)} tasks')
    print(f'\nwrote tables and figures to {out_dir}')


if __name__ == '__main__':
    main()
