"""Build the preliminary-results slide deck from the scored results and the human audit.

Every number on the slides is computed from:
    results/scored/{dev, dev_nearmiss, test_robust}.csv
    results/analysis/test_robust/{summary, effort_effects}.csv
    results/audit/distractor_audit.csv
Output: results/presentation/webshop_preliminary_results.pptx (+ figures/)

Usage (repo root; pandas, matplotlib, python-pptx):
    python experiments/make_presentation.py
"""
import os
from os.path import abspath, dirname, join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

REPO_DIR = dirname(dirname(abspath(__file__)))
RESULTS = join(REPO_DIR, 'results')
OUT_DIR = join(RESULTS, 'presentation')
FIG_DIR = join(OUT_DIR, 'figures')

MODELS = ['gpt-6.1-sol', 'gemini-3.1-pro', 'claude-opus-5.5']
NAMES = {'gpt-6.1-sol': 'GPT-6.1 Sol', 'gemini-3.1-pro': 'Gemini 3.1 Pro', 'claude-opus-5.5': 'Claude Opus 5.5'}
COLORS = {'gpt-6.1-sol': '#2a9d8f', 'gemini-3.1-pro': '#e76f51', 'claude-opus-5.5': '#264653'}
EFFORTS = ['low', 'medium', 'high']

NAVY = RGBColor(0x1d, 0x35, 0x57)
ACCENT = RGBColor(0xe7, 0x6f, 0x51)
GREY = RGBColor(0x55, 0x5b, 0x66)


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def load():
    data = {
        'test': pd.read_csv(join(RESULTS, 'scored', 'test_robust.csv')),
        'pilot_bm25': pd.read_csv(join(RESULTS, 'scored', 'dev.csv')),
        'pilot_nearmiss': pd.read_csv(join(RESULTS, 'scored', 'dev_nearmiss.csv')),
        'summary': pd.read_csv(join(RESULTS, 'analysis', 'test_robust', 'summary.csv')),
        'effects': pd.read_csv(join(RESULTS, 'analysis', 'test_robust', 'effort_effects.csv')),
        'audit': pd.read_csv(join(RESULTS, 'audit', 'distractor_audit.csv'), encoding='utf-8-sig'),
    }
    data['test']['valid'] = data['test']['valid'].astype(bool)
    return data


def compute_stats(d):
    test, summary, effects, audit = d['test'], d['summary'], d['effects'], d['audit']
    v = test[test['valid']].copy()
    v['success'] = v['success'].astype(bool)
    v['success_lenient'] = v['success_lenient'].astype(bool)
    s = {'n_tasks': test['task_id'].nunique(), 'n_responses': len(test), 'n_valid': int(test['valid'].sum())}

    s['summary'] = summary.set_index(['model_key', 'effort'])
    hl = effects[effects['contrast'] == 'high - low'].set_index(['model_key', 'metric'])
    s['effects'] = hl

    s['same_choice_all_efforts'] = {}
    wide = v.pivot_table(index='task_id', columns=['model_key', 'effort'], values='chosen_asin', aggfunc='first')
    for m in MODELS:
        sub = wide[m].dropna()
        s['same_choice_all_efforts'][m] = (sub.nunique(axis=1) == 1).mean()

    s['conf_right_wrong'] = v.groupby(['model_key', 'success'])['confidence'].mean().unstack()
    s['share_conf10'] = v.assign(c10=v['confidence'] == 10).groupby('model_key')['c10'].mean()

    low = v[v['effort'] == 'low']
    wrong = low[~low['success']]
    per = wrong.groupby('task_id').agg(n_models=('model_key', 'nunique'), n_products=('chosen_asin', 'nunique'))
    s['conc_any_wrong'] = len(per)
    s['conc_all_wrong'] = int((per['n_models'] == 3).sum())
    s['conc_same_wrong'] = int(((per['n_models'] == 3) & (per['n_products'] == 1)).sum())

    strict_vs_lenient = (v.groupby(['model_key', 'effort'])['success'].mean()
                         - v.groupby(['model_key', 'effort'])['success_lenient'].mean()).abs()
    s['max_strict_lenient_gap'] = strict_vs_lenient.max()

    verdicts = audit['violates_instruction'].str.strip().str.lower()
    s['audit_n'] = len(audit)
    s['audit_violates'] = int((verdicts == 'y').sum())
    s['audit_false'] = int((verdicts == 'n').sum())
    s['audit_ci'] = wilson(s['audit_false'], s['audit_n'])

    pb = d['pilot_bm25'].groupby(['model_key', 'effort'])['success'].mean()
    s['pilot_bm25_range'] = (pb.min(), pb.max())
    pn = d['pilot_nearmiss'].groupby(['model_key', 'effort']).agg(
        strict=('success', 'mean'), lenient=('success_lenient', 'mean'))
    s['pilot_nm_range'] = (pn['strict'].min(), pn['strict'].max())
    s['pilot_nm_lenient_gain'] = (pn['lenient'] - pn['strict']).mean()
    return s


def fig_effort(summary, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    for m in MODELS:
        g = summary.loc[m].reindex(EFFORTS)
        x = np.arange(3)
        axes[0].errorbar(x, g['success'] * 100, yerr=[(g['success'] - g['success_ci_low']) * 100,
                                                    (g['success_ci_high'] - g['success']) * 100],
                         marker='o', capsize=4, color=COLORS[m], label=NAMES[m], linewidth=2)
        axes[1].plot(x, g['mean_confidence'], marker='o', color=COLORS[m], linewidth=2)
        axes[2].plot(x, g['mean_reasoning_tokens'], marker='o', color=COLORS[m], linewidth=2)
    titles = ['Correctness: success rate (%)', 'Stated confidence (0-10)', 'Reasoning tokens used (mean)']
    for ax, title in zip(axes, titles):
        ax.set_xticks(range(3))
        ax.set_xticklabels(EFFORTS)
        ax.set_xlabel('reasoning effort setting')
        ax.set_title(title, fontsize=12)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylim(50, 100)
    axes[1].set_ylim(5, 10)
    axes[0].legend(frameon=False, fontsize=10, loc='lower left')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_calibration(summary, path):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([50, 100], [50, 100], color='grey', linestyle='--', linewidth=1)
    ax.text(88.5, 92, 'perfectly calibrated', color='grey', fontsize=9, rotation=45, ha='center')
    sizes = {'low': 50, 'medium': 110, 'high': 190}
    for m in MODELS:
        g = summary.loc[m].reindex(EFFORTS)
        ax.plot(g['mean_confidence'] * 10, g['success'] * 100, color=COLORS[m], linewidth=1, alpha=0.6)
        for e in EFFORTS:
            ax.scatter(g.loc[e, 'mean_confidence'] * 10, g.loc[e, 'success'] * 100, s=sizes[e], color=COLORS[m],
                       edgecolor='white', zorder=3)
        offset = (12, 10) if m == 'claude-opus-5.5' else (-12, 10)
        ax.annotate(NAMES[m], (g['mean_confidence'].iloc[-1] * 10, g['success'].iloc[-1] * 100),
                    textcoords='offset points', xytext=offset, fontsize=10, color=COLORS[m],
                    ha='left' if m == 'claude-opus-5.5' else 'right')
    ax.set_xlim(70, 100)
    ax.set_ylim(70, 100)
    ax.set_xlabel('mean stated confidence (%)')
    ax.set_ylabel('actual success rate (%)')
    ax.set_title('Below the diagonal = overconfident\n(dot size: low / medium / high effort)', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_right_wrong(conf_rw, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    x = np.arange(len(MODELS))
    right = [conf_rw.loc[m, True] for m in MODELS]
    wrong = [conf_rw.loc[m, False] for m in MODELS]
    ax.bar(x - 0.18, right, 0.36, label='when correct', color='#8ab17d')
    ax.bar(x + 0.18, wrong, 0.36, label='when wrong', color='#e76f51')
    for i in range(len(MODELS)):
        ax.text(x[i] - 0.18, right[i] + 0.05, f'{right[i]:.1f}', ha='center', fontsize=10)
        ax.text(x[i] + 0.18, wrong[i] + 0.05, f'{wrong[i]:.1f}', ha='center', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[m] for m in MODELS])
    ax.set_ylim(5, 11)
    ax.set_ylabel('mean stated confidence (0-10)')
    ax.set_title('Does confidence drop when the model is wrong?', fontsize=11)
    ax.legend(frameon=False, loc='upper center', ncol=2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_concentration(s, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = ['At least one\nmodel wrong', 'All three\nmodels wrong', 'All three picked the\nsame wrong product']
    values = [s['conc_any_wrong'], s['conc_all_wrong'], s['conc_same_wrong']]
    bars = ax.bar(labels, values, color=['#a8dadc', '#457b9d', '#1d3557'])
    for b, val in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.5, str(val), ha='center', fontsize=12)
    ax.set_ylabel(f'number of tasks (of {s["n_tasks"]})')
    ax.set_title('Errors are shared across models (low effort)', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.333)
        self.prs.slide_height = Inches(7.5)
        self.blank = self.prs.slide_layouts[6]

    def slide(self, title, notes=''):
        sl = self.prs.slides.add_slide(self.blank)
        bar = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, self.prs.slide_width, Inches(0.12))
        bar.fill.solid()
        bar.fill.fore_color.rgb = NAVY
        bar.line.fill.background()
        self.text(sl, title, Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.9), size=30, bold=True, color=NAVY)
        if notes:
            sl.notes_slide.notes_text_frame.text = notes
        return sl

    def text(self, sl, text, left, top, width, height, size=18, bold=False, color=GREY, align=None):
        tb = sl.shapes.add_textbox(left, top, width, height)
        tf = tb.text_frame
        tf.word_wrap = True
        run = tf.paragraphs[0].add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        if align is not None:
            tf.paragraphs[0].alignment = align
        return tb

    def bullets(self, sl, items, left, top, width, height, size=18):
        tf = sl.shapes.add_textbox(left, top, width, height).text_frame
        tf.word_wrap = True
        for i, item in enumerate(items):
            level = 1 if item.startswith('  ') else 0
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            run = p.add_run()
            run.text = ('– ' if level else '• ') + item.strip()
            run.font.size = Pt(size - 2 * level)
            run.font.color.rgb = GREY if level else RGBColor(0x22, 0x22, 0x22)
            p.level = level
            p.space_after = Pt(8)
        return tf

    def table(self, sl, rows, left, top, width, height, col_widths=None, size=14):
        shape = sl.shapes.add_table(len(rows), len(rows[0]), left, top, width, height)
        table = shape.table
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                cell = table.cell(r, c)
                cell.text = str(val)
                para = cell.text_frame.paragraphs[0]
                para.alignment = PP_ALIGN.CENTER if c else PP_ALIGN.LEFT
                for run in para.runs:
                    run.font.size = Pt(size)
                    run.font.bold = r == 0
        if col_widths:
            for c, w in enumerate(col_widths):
                table.columns[c].width = Inches(w)
        return table

    def callout(self, sl, text, left, top, width, height, size=17):
        box = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0xf1, 0xf4, 0xf8)
        box.line.color.rgb = ACCENT
        tf = box.text_frame
        tf.word_wrap = True
        run = tf.paragraphs[0].add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.color.rgb = NAVY
        run.font.bold = True


def pct(x):
    return f'{100 * x:.0f}%'


def build(d, s, figs, out_path):
    sm, ef = s['summary'], s['effects']
    deck = Deck()

    # 1. Title
    sl = deck.prs.slides.add_slide(deck.blank)
    bg = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, deck.prs.slide_width, deck.prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    deck.text(sl, 'Does Thinking Make AI Shopping Agents More Correct, or Just More Confident?',
              Inches(0.9), Inches(2.2), Inches(11.5), Inches(1.8), size=38, bold=True, color=RGBColor(255, 255, 255))
    deck.text(sl, 'Frontier models on the WebShop benchmark: preliminary results',
              Inches(0.9), Inches(4.1), Inches(11.5), Inches(0.8), size=22, color=RGBColor(0xd8, 0xe2, 0xec))
    deck.text(sl, f'{s["n_tasks"]} test instructions  |  3 models  |  3 reasoning-effort levels  |  '
                  f'{s["n_responses"]:,} model decisions',
              Inches(0.9), Inches(5.0), Inches(11.5), Inches(0.6), size=16, color=RGBColor(0xd8, 0xe2, 0xec))

    # 2. Why WebShop
    sl = deck.slide('Why WebShop: correctness we can measure objectively', notes=(
        'ACES could only tell us which products models liked, not whether they were right. '
        'WebShop instructions were written by crowd workers looking at a specific product, so every '
        'instruction has stated requirements we can check.'))
    deck.bullets(sl, [
        'ACES (Allouah et al., 2025) has no intrinsic ground truth: no product is objectively "correct"',
        'WebShop (Yao et al., 2022): 1.18M real Amazon products and 12,087 crowd-written shopping instructions',
        '  e.g. "purple, machine-washable drapes, 52-inch width, under $60"',
        'Every instruction has checkable requirements: product type, features, options (size/color), price ceiling',
        'WebShop reward = product type match x share of requirements met; success = reward of exactly 1.0',
        'We keep the research questions from the ACES study: reasoning effort, cross-model comparison, '
        'calibration, and market concentration',
    ], Inches(0.7), Inches(1.4), Inches(12), Inches(5.5), size=20)

    # 3. Design
    sl = deck.slide('Experimental design', notes=(
        'Choice-set format mirrors ACES so findings are comparable, but now with exactly one correct answer. '
        'Instructions are the first 150 of WebShop\'s standard 500-instruction test split.'))
    deck.bullets(sl, [
        f'{s["n_tasks"]} instructions from WebShop\'s standard human-written test split',
        'Each task shows 10 products (title, price, features, description, options, and product image)',
        '  exactly 1 product satisfies every requirement',
        '  9 "near-miss" distractors of the same kind of product, each failing at least one requirement',
        '  same products and order for every model, so choices are directly comparable',
        'The model returns: chosen product, option values (e.g. size, color), and confidence from 0 to 10',
        'Models (via OpenRouter): GPT-6.1 Sol, Claude Opus 5.5, Gemini 3.1 Pro',
        'Reasoning effort: low / medium / high, with an identical output budget across all conditions',
        'Catalog: 100,000 products, including every product with a human-written instruction',
    ], Inches(0.7), Inches(1.4), Inches(12), Inches(5.6), size=19)

    # 4. Ground truth
    lo_ci, hi_ci = s['audit_ci']
    sl = deck.slide('Making sure "correct" really means correct', notes=(
        'This addresses the main concern with ACES. Each pilot exposed a problem that we fixed before the main run. '
        'The final audit was AI-assisted and reviewed by hand: every one of the 50 sampled distractors was checked '
        'against its instruction.'))
    rows = [
        ['Step', 'What we found', 'What we changed'],
        ['Pilot 1: search-ranked distractors',
         f'Success {pct(s["pilot_bm25_range"][0])}-{pct(s["pilot_bm25_range"][1])}: distractors too easy',
         'Use near-miss distractors'],
        ['Pilot 2: near-miss distractors',
         'Many "errors" were scoring artifacts (e.g. "non-slip" vs "non slip", 52 x 54 inch vs 52"w x 54"l)',
         'Drop distractors that pass a lenient check; score both ways'],
        ['Main run (this deck)',
         f'Strict and lenient scoring differ by at most {100 * s["max_strict_lenient_gap"]:.1f} points',
         'Human audit of 50 random distractors'],
    ]
    deck.table(sl, rows, Inches(0.6), Inches(1.45), Inches(12.1), Inches(2.6), col_widths=[3.0, 5.6, 3.5], size=14)
    deck.callout(sl, f'Human-reviewed audit: {s["audit_violates"]} of {s["audit_n"]} distractors genuinely violate '
                     f'their instruction. Residual error rate {pct(s["audit_false"] / s["audit_n"])} '
                     f'(95% CI {pct(lo_ci)}-{pct(hi_ci)}). Most causes (options not checked, unit conversions, '
                     f'a text-normalization bug) are being fixed; synonyms remain the main residual.',
                 Inches(0.6), Inches(4.65), Inches(12.1), Inches(1.4), size=17)

    # 5. Main result
    sl = deck.slide('More thinking raises confidence, not correctness', notes=(
        'Left: success is flat across effort levels for all three models. Middle: confidence is flat for Claude '
        'but creeps up for GPT and especially Gemini. Right: only Gemini substantially increases its reasoning '
        'when asked; GPT and Claude barely reason on this task at any setting.'))
    sl.shapes.add_picture(figs['effort'], Inches(0.3), Inches(1.3), width=Inches(12.7))
    deck.text(sl, 'Error bars: 95% confidence intervals. Same 150 tasks in every condition.',
              Inches(0.6), Inches(5.4), Inches(12), Inches(0.4), size=12)
    deck.bullets(sl, [
        f'GPT-6.1 Sol picks the same product at all three effort levels on '
        f'{pct(s["same_choice_all_efforts"]["gpt-6.1-sol"])} of tasks; Claude on '
        f'{pct(s["same_choice_all_efforts"]["claude-opus-5.5"])} and Gemini on '
        f'{pct(s["same_choice_all_efforts"]["gemini-3.1-pro"])}',
        'Only Gemini substantially changes how much it reasons when effort is raised',
    ], Inches(0.6), Inches(5.85), Inches(12.2), Inches(1.4), size=16)

    # 6. Effort effects table
    def eff(m, metric, scale, decimals):
        r = ef.loc[(m, metric)]
        vals = [round(scale * r[k], decimals) + 0.0 for k in ['diff', 'ci_low', 'ci_high']]
        return f'{vals[0]:+.{decimals}f}  [{vals[1]:+.{decimals}f}, {vals[2]:+.{decimals}f}]'
    sl = deck.slide('Effect of raising effort from low to high (paired, same tasks)', notes=(
        'Paired differences over the same 150 tasks with bootstrap 95% intervals. '
        'Gemini is the clearest case of the hypothesis: confidence rises reliably while the gain in success is '
        'small and its interval touches zero. GPT shows exactly zero change in success with a small confidence increase.'))
    rows = [['Model', 'Change in success (pts)', 'Change in confidence (0-10)', 'Reasoning tokens (low → high)']]
    for m in MODELS:
        rows.append([NAMES[m], eff(m, 'success', 100, 1), eff(m, 'confidence', 1, 2),
                     f'{sm.loc[(m, "low"), "mean_reasoning_tokens"]:.0f} → {sm.loc[(m, "high"), "mean_reasoning_tokens"]:.0f}'])
    deck.table(sl, rows, Inches(0.6), Inches(1.5), Inches(12.1), Inches(2.2), col_widths=[2.6, 3.4, 3.4, 2.7], size=16)
    deck.bullets(sl, [
        'GPT-6.1 Sol: correctness unchanged; confidence rises slightly but reliably',
        'Gemini 3.1 Pro: confidence rises clearly; the small gain in success is not statistically reliable',
        'Claude Opus 5.5: neither correctness nor confidence changes',
        'Values in brackets: bootstrap 95% confidence intervals',
    ], Inches(0.6), Inches(4.2), Inches(12.1), Inches(2.6), size=18)

    # 7. Calibration
    sl = deck.slide('Calibration: the OpenAI vs. Claude gap from ACES replicates', notes=(
        'Overconfidence = mean stated confidence (as a fraction) minus actual success. '
        'AUROC measures whether confidence separates right from wrong answers: 0.5 is chance, 1.0 is perfect.'))
    sl.shapes.add_picture(figs['calibration'], Inches(0.4), Inches(1.4), height=Inches(5.0))
    def span(values, fmt, suffix=''):
        lo, hi = format(values.min(), fmt), format(values.max(), fmt)
        return f'{lo}{suffix}' if lo == hi else f'{lo}-{hi}{suffix}'
    rows = [['Model', 'Success', 'Confidence', 'Over-confidence', 'Answers at 10/10', 'AUROC*']]
    for m in MODELS:
        g = sm.loc[m]
        rows.append([NAMES[m], span(100 * g['success'], '.0f', '%'), span(g['mean_confidence'], '.1f'),
                     f'{100 * g["overconfidence"].mean():+.0f} pts', pct(s['share_conf10'][m]),
                     span(g['auroc'], '.2f')])
    deck.table(sl, rows, Inches(6.5), Inches(1.5), Inches(6.6), Inches(2.2),
               col_widths=[1.45, 0.8, 1.1, 1.35, 0.9, 1.0], size=12)
    deck.text(sl, '*AUROC: how well confidence separates right from wrong answers (0.5 = chance, 1.0 = perfect). '
                  'Ranges span the three effort levels.',
              Inches(6.5), Inches(3.8), Inches(6.6), Inches(0.6), size=11)
    deck.bullets(sl, [
        'GPT and Gemini: near-maximal confidence, roughly 15 points overconfident',
        'Claude: lower confidence, similar or higher accuracy, slightly underconfident',
    ], Inches(6.5), Inches(4.6), Inches(6.6), Inches(2.2), size=16)

    # 8. Confidence right vs wrong
    crw = s['conf_right_wrong']
    sl = deck.slide("Only Claude's confidence tells you when it is wrong", notes=(
        'Pooled over all effort levels. GPT reports essentially the same confidence whether right or wrong; '
        'Claude drops by nearly a full point.'))
    sl.shapes.add_picture(figs['right_wrong'], Inches(0.4), Inches(1.4), height=Inches(5.0))
    deck.bullets(sl, [
        f'Claude: {crw.loc["claude-opus-5.5", True]:.1f} when correct vs {crw.loc["claude-opus-5.5", False]:.1f} when wrong',
        f'Gemini: {crw.loc["gemini-3.1-pro", True]:.1f} vs {crw.loc["gemini-3.1-pro", False]:.1f}',
        f'GPT: {crw.loc["gpt-6.1-sol", True]:.1f} vs {crw.loc["gpt-6.1-sol", False]:.1f} '
        '(slightly more confident when wrong)',
        'For Gemini, confidence becomes less informative as effort increases '
        f'(AUROC {sm.loc[("gemini-3.1-pro", "low"), "auroc"]:.2f} → {sm.loc[("gemini-3.1-pro", "high"), "auroc"]:.2f})',
    ], Inches(7.5), Inches(1.7), Inches(5.4), Inches(4.8), size=17)

    # 9. Concentration
    sl = deck.slide('Market concentration: models fail together', notes=(
        'Low-effort condition. When models make mistakes, they tend to make the same mistake: '
        'in most tasks where any model erred, all three erred, usually picking the identical wrong product.'))
    sl.shapes.add_picture(figs['concentration'], Inches(0.4), Inches(1.4), height=Inches(4.6))
    deck.bullets(sl, [
        f'Of {s["conc_any_wrong"]} tasks where at least one model chose wrong, all three chose wrong on '
        f'{s["conc_all_wrong"]}',
        f'On {s["conc_same_wrong"]} of those, all three picked the same wrong product',
        'Errors are correlated rather than independent: switching between these models offers '
        'little protection against a bad recommendation',
    ], Inches(7.5), Inches(1.7), Inches(5.4), Inches(4.8), size=17)

    # 10. Caveats
    sl = deck.slide('Caveats', notes='Limitations to keep in mind when interpreting the preliminary results.')
    deck.bullets(sl, [
        'Weak manipulation for GPT and Claude: they use very few reasoning tokens at every effort setting, '
        'so for them the finding is "the effort setting barely changes behavior on this task"',
        f'Residual ground-truth noise: about {pct(s["audit_false"] / s["audit_n"])} of distractors may actually be '
        'acceptable, so some measured errors are false negatives',
        'Some distractors are an obviously different kind of product, which makes those tasks easy',
        f'Preliminary scale: {s["n_tasks"]} tasks and 3 models; differences of a few points between models '
        'are not yet reliable',
        'Fixed choice sets, not the full interactive WebShop task (searching and navigating pages)',
    ], Inches(0.7), Inches(1.4), Inches(12), Inches(5.6), size=19)

    # 11. Next steps
    sl = deck.slide('Next steps', notes='Plan for the full study.')
    deck.bullets(sl, [
        'Fix the audit-identified issues: check product options, handle unit conversions and synonyms; '
        're-audit a fresh sample',
        'Tighten distractors to the same product category so every task is a genuine near miss',
        'Full study: all 496 test instructions x 10 frontier models x 3 effort levels',
        '  OpenAI: GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna;  Anthropic: Claude Fable 5.1, Claude Opus 5.5;  '
        'Google: Gemini 3.1 Pro',
        '  Others: DeepSeek V4 Pro, Qwen 3.8 Max, Kimi K3, GLM-5V Turbo',
        'Test whether harder tasks (e.g. larger choice sets) make models actually use more reasoning',
        'Robustness: interactive WebShop agent (search + navigation) on a subset',
    ], Inches(0.7), Inches(1.4), Inches(12), Inches(5.6), size=19)

    deck.prs.save(out_path)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    d = load()
    s = compute_stats(d)
    figs = {
        'effort': join(FIG_DIR, 'effort.png'),
        'calibration': join(FIG_DIR, 'calibration.png'),
        'right_wrong': join(FIG_DIR, 'confidence_right_wrong.png'),
        'concentration': join(FIG_DIR, 'concentration.png'),
    }
    fig_effort(s['summary'], figs['effort'])
    fig_calibration(s['summary'], figs['calibration'])
    fig_right_wrong(s['conf_right_wrong'], figs['right_wrong'])
    fig_concentration(s, figs['concentration'])
    out_path = join(OUT_DIR, 'webshop_preliminary_results.pptx')
    build(d, s, figs, out_path)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
