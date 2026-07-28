"""
Generate a professional multi-part PDF validation report from a Surrogate Factory
metadata JSON.

Document structure
------------------
  Part 1  Executive Summary  (≤ 2 pages) — winner model only.
  Part 2  Model Comparison   — all models, side-by-side tables.
  ─── separator page ───
  Part 3  Technical Review   — full report mirroring the HTML validation output:
          TOC (own page) → Introduction → Accuracy → Scatter plots →
          Ratio-error plots → KS overfitting → Split quality →
          Improvement roadmap.

Usage
-----
    python generate_executive_summary.py <metadata.json> [--output <dir>]
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ── LaTeX helpers ──────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    for old, new in [
        ('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
        ('$', r'\$'), ('#', r'\#'), ('_', r'\_'),
        ('{', r'\{'), ('}', r'\}'),
        ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}'),
    ]:
        s = s.replace(old, new)
    return s


def _pct(v, decimals=0):
    if v is None:
        return r'\textemdash'
    return rf'{v * 100:.{decimals}f}\%'


def _r2cell(r2):
    if r2 is None:
        return r'\textemdash'
    if r2 >= 0.95:
        return rf'\cellcolor{{r2green}}{r2:.4f}'
    if r2 >= 0.80:
        return rf'\cellcolor{{r2amber}}{r2:.4f}'
    return rf'\cellcolor{{failred}}{r2:.4f}'


def _q90cell(val, passed):
    if val is None:
        return r'\textemdash'
    color = 'passgreen' if passed else 'failred'
    return rf'\cellcolor{{{color}}}{val:.4f}'


def _gapcell(gap):
    if gap is None:
        return r'\textemdash'
    sign = '+' if gap > 0 else ''
    color = 'failred' if gap > 0 else 'passgreen'
    return rf'\cellcolor{{{color}}}{sign}{gap:.4f}'


def _kscell(pval):
    if pval is None:
        return r'\textemdash'
    color = 'passgreen' if pval >= 0.05 else 'failred'
    return rf'\cellcolor{{{color}}}{pval:.3f}'


# ── Data extraction ────────────────────────────────────────────────────────────

def extract(meta_path: Path) -> dict:
    with open(meta_path) as f:
        root = json.load(f)['metadata']

    val  = root['Model_Validation']
    sel  = root['Model_Selection']
    trn  = root.get('Model_Training', {})
    part = (root.get('Data_Partition', {})
                .get('data_split', {})
                .get('percentages', {}))
    req  = root.get('Requirements', {})

    models  = [m['label'] for m in trn.get('Models', [])]
    outputs = sel.get('outputs', [])
    inputs  = sel.get('inputs', [])
    scores  = val.get('scores', {})
    dist    = val.get('distribution_tests', {})
    vres    = val.get('validation_results', [])
    split   = val.get('split_validation', {})
    acc_list   = req.get('accuracy', [])
    q90_target = acc_list[0]['value'] if acc_list else 0.10

    avg_r2 = {}
    for mdl in models:
        r2s = [scores.get(mdl, {}).get(o, {}).get('R2') for o in outputs]
        r2s = [v for v in r2s if v is not None]
        avg_r2[mdl] = sum(r2s) / len(r2s) if r2s else 0

    best = max(avg_r2, key=avg_r2.get) if avg_r2 else (models[0] if models else 'N/A')

    q90_pass = {m: 0 for m in models}
    for vr in vres:
        for mdl, res in vr.get('models', {}).items():
            if res.get('passed'):
                q90_pass[mdl] = q90_pass.get(mdl, 0) + 1

    ks_pass = {mdl: sum(1 for v in dist.get(mdl, {}).values() if v >= 0.05)
               for mdl in models}

    return dict(
        models=models, outputs=outputs, inputs=inputs,
        scores=scores, dist=dist, vres=vres, split=split,
        q90_target=q90_target, best_model=best,
        avg_r2=avg_r2, q90_pass=q90_pass, ks_pass=ks_pass,
        pct_train=part.get('train'),
        pct_val=part.get('validation') or part.get('val'),
        pct_test=part.get('test'),
        use_case=meta_path.stem.replace('metadata_', ''),
    )


# ── Per-output row list ────────────────────────────────────────────────────────

def _per_output(d, model):
    vres_map = {vr['output']: vr for vr in d['vres']}
    rows = []
    for o in d['outputs']:
        mres = vres_map.get(o, {}).get('models', {}).get(model, {})
        r2   = d['scores'].get(model, {}).get(o, {}).get('R2')
        q90  = mres.get('score')
        ok   = mres.get('passed', False)
        gap  = round(q90 - d['q90_target'], 6) if q90 is not None else None
        rows.append(dict(output=o, r2=r2, q90=q90, passed=ok, gap=gap))
    return rows


# ── Recommendation bullets ─────────────────────────────────────────────────────

def _recommendations(d, model, rows, detailed=False):
    target   = d['q90_target']
    failed   = [r for r in rows if not r['passed']]
    low_r2   = [r for r in rows if r['r2'] is not None and r['r2'] < 0.80]
    ks_fails = [o for o in d['outputs']
                if (d['dist'].get(model, {}).get(o) or 1.0) < 0.05]
    n_out    = len(d['outputs'])
    bullets  = []

    if not failed:
        bullets.append(
            r'\textbf{Production-ready.} All outputs satisfy the Q90 accuracy '
            r'requirement. The model can be deployed with confidence.'
        )
    else:
        slight = [r for r in failed if r['gap'] is not None and r['gap'] < 0.05]
        severe = [r for r in failed if r['gap'] is not None and r['gap'] >= 0.05]
        if slight:
            names = ', '.join(_esc(r['output']) for r in slight)
            extra = (r' Collect additional data near under-represented '
                     r'flight conditions for these outputs.' if detailed else '')
            bullets.append(
                rf'\textbf{{Marginal failures}} (\textit{{{names}}}): '
                rf'Q90 within 5\% of target. Targeted data augmentation likely sufficient.{extra}'
            )
        if severe:
            names = ', '.join(_esc(r['output']) for r in severe)
            extra = (r' Review whether additional physical drivers are missing '
                     r'from the input feature set, or consider a more expressive '
                     r'architecture (deeper network, ensemble).' if detailed else '')
            bullets.append(
                rf'\textbf{{Significant failures}} (\textit{{{names}}}): '
                rf'Q90 exceeds target by $\geq 5\%$. '
                rf'Investigate input features and model capacity.{extra}'
            )

    if low_r2:
        names = ', '.join(_esc(r['output']) for r in low_r2)
        extra = (r' A low R\textsuperscript{2} often indicates missing physical '
                 r'drivers or highly nonlinear behaviour requiring richer features.'
                 if detailed else '')
        bullets.append(
            rf'\textbf{{Low R\textsuperscript{{2}}}} (\textit{{{names}}}): '
            rf'Below 0.80 --- high unexplained variance.{extra}'
        )

    if ks_fails:
        names = ', '.join(_esc(o) for o in ks_fails)
        extra = (r' Increase regularisation (dropout, weight decay, '
                 r'early-stopping patience) or reduce model capacity.' if detailed else '')
        bullets.append(
            rf'\textbf{{Overfitting detected}} (\textit{{{names}}}): '
            rf'KS test rejects equal residual distributions on train and test.{extra}'
        )
    else:
        bullets.append(
            r'\textbf{No overfitting.} KS test confirms consistent residual '
            r'distributions on both train and test sets.'
        )

    return bullets


# ── LaTeX preamble ─────────────────────────────────────────────────────────────

_PREAMBLE = r"""\documentclass[9pt,a4paper]{article}
\usepackage[margin=1.8cm,top=2.0cm,bottom=2.0cm]{geometry}
\usepackage{booktabs,colortbl,xcolor,array,makecell,hhline}
\usepackage{amsmath,amssymb}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{fancyhdr,graphicx}
\usepackage[colorlinks=true,linkcolor=primary,urlcolor=primary,
            bookmarks=true,pdfborder={0 0 0}]{hyperref}
\usepackage{parskip,enumitem,titlesec}
\setlength{\parskip}{2pt}

% ── Colours ───────────────────────────────────────────────────────────────────
\definecolor{passgreen}{RGB}{198,239,206}
\definecolor{failred}{RGB}{255,199,206}
\definecolor{warnamber}{RGB}{255,235,156}
\definecolor{r2green}{RGB}{198,239,206}
\definecolor{r2amber}{RGB}{255,235,156}
\definecolor{primary}{RGB}{0,70,127}
\definecolor{sepgray}{RGB}{130,130,130}

% ── Section style ─────────────────────────────────────────────────────────────
\titleformat{\section}{\normalsize\bfseries\color{primary}}{}{0em}{}
            [\vspace{-4pt}\textcolor{primary}{\rule{\linewidth}{0.5pt}}]
\titlespacing{\section}{0pt}{10pt}{4pt}

% ── Lightweight heading (safe inside minipage) ────────────────────────────────
\newcommand{\exhead}[1]{%
  \medskip\noindent{\small\bfseries\color{primary}#1}%
  \par\vspace{-5pt}\noindent\textcolor{primary}{\rule{\linewidth}{0.4pt}}\vspace{2pt}}

% ── Page style ────────────────────────────────────────────────────────────────
\pagestyle{fancy}\fancyhf{}
\lhead{\small\textcolor{primary}{\textbf{Surrogate Factory}}}
\chead{\small\textcolor{sepgray}{CONFIDENTIAL}}
\rhead{\small\textcolor{primary}{XUSECASEX}}
\lfoot{\small\textcolor{sepgray}{Surrogate Factory v2.2 --- Automated Validation Report}}
\rfoot{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}
"""


# ── Banner (Part 1) ────────────────────────────────────────────────────────────

def _banner(d, best, rows):
    """Verdict banner: 3 colour-coded rows (R², Q90, KS) with output names."""
    n_out     = len(d['outputs'])
    best_r2   = d['avg_r2'].get(best, 0)
    best_q90p = d['q90_pass'].get(best, 0)
    best_ksp  = d['ks_pass'].get(best, 0)

    overall_ok  = (best_q90p == n_out) and (best_ksp == n_out)
    box_color   = 'passgreen' if overall_ok else 'warnamber'
    verdict_txt = 'ALL REQUIREMENTS MET' if overall_ok else 'REQUIREMENTS PARTIALLY MET'

    # R² row
    if best_r2 >= 0.95:
        r2_color, r2_label = 'r2green',  'Excellent'
    elif best_r2 >= 0.80:
        r2_color, r2_label = 'r2amber',  'Acceptable'
    else:
        r2_color, r2_label = 'failred',  'Poor --- needs attention'
    r2_str = f'{best_r2:.4f}'

    # Q90 row — list which outputs pass/fail
    q90_passed_names = [_esc(r['output']) for r in rows if r['passed']]
    q90_failed_names = [_esc(r['output']) for r in rows if not r['passed']]
    if best_q90p == n_out:
        q90_color  = 'passgreen'
        q90_label  = 'All outputs pass'
        q90_detail = ', '.join(q90_passed_names)
        q90_fail   = ''
    elif best_q90p > 0:
        q90_color  = 'warnamber'
        q90_label  = 'Partially met'
        q90_detail = 'Pass: ' + ', '.join(q90_passed_names)
        q90_fail   = 'Fail: ' + ', '.join(q90_failed_names)
    else:
        q90_color  = 'failred'
        q90_label  = 'No output meets requirement'
        q90_detail = ''
        q90_fail   = 'Fail: ' + ', '.join(q90_failed_names)

    # KS row — list which outputs pass/fail
    ks_passed_names = [_esc(o) for o in d['outputs']
                       if (d['dist'].get(best, {}).get(o) or 1.0) >= 0.05]
    ks_failed_names = [_esc(o) for o in d['outputs']
                       if (d['dist'].get(best, {}).get(o) or 1.0) < 0.05]
    if best_ksp == n_out:
        ks_color  = 'passgreen'
        ks_label  = 'No overfitting detected'
        ks_detail = ', '.join(ks_passed_names)
        ks_fail   = ''
    elif best_ksp > n_out // 2:
        ks_color  = 'warnamber'
        ks_label  = 'Minor overfitting'
        ks_detail = 'Pass: ' + ', '.join(ks_passed_names)
        ks_fail   = 'Fail: ' + ', '.join(ks_failed_names)
    else:
        ks_color  = 'failred'
        ks_label  = 'Overfitting present'
        ks_detail = ('Pass: ' + ', '.join(ks_passed_names)) if ks_passed_names else ''
        ks_fail   = 'Fail: ' + ', '.join(ks_failed_names)

    best_esc = _esc(best)

    def _metric_row(color, title, label, detail, fail):
        lines = [rf'{{\small\bfseries {title}}} \enspace---\enspace {{\small {label}}}']
        if detail:
            lines.append(rf'{{\footnotesize {detail}}}')
        if fail:
            lines.append(rf'{{\footnotesize\color{{red}} {fail}}}')
        body = r'\\[1pt]'.join(lines)
        return (
            rf'\colorbox{{{color}}}{{\begin{{minipage}}{{0.97\linewidth}}'
            rf'\vspace{{2pt}}\centering {body}\vspace{{2pt}}\end{{minipage}}}}'
        )

    r2_row  = _metric_row(r2_color,  rf'Avg R\textsuperscript{{2}}: {r2_str}',
                          r2_label, '', '')
    q90_row = _metric_row(q90_color, 'Q90 Accuracy', q90_label, q90_detail, q90_fail)
    ks_row  = _metric_row(ks_color,  'Overfitting check (KS)', ks_label, ks_detail, ks_fail)

    return rf"""\begin{{center}}
\colorbox{{{box_color}}}{{\begin{{minipage}}{{0.66\linewidth}}
  \vspace{{5pt}}\centering
  {{\normalsize\bfseries {verdict_txt}}}\\[4pt]
  {{\small\bfseries Recommended model:}} {{\small {best_esc}}}\\[5pt]
  {r2_row}\\[2pt]
  {q90_row}\\[2pt]
  {ks_row}
  \vspace{{5pt}}
\end{{minipage}}}}
\end{{center}}"""


# ── Part 1: Executive Summary ──────────────────────────────────────────────────

def _part1(d, best, rows, recs, scatter_paths):
    uc          = _esc(d['use_case'])
    n_out       = len(d['outputs'])
    n_models    = len(d['models'])
    date_str    = datetime.today().strftime('%d %B %Y')
    inputs_str  = ', '.join(_esc(i) for i in d['inputs'])
    outputs_str = ', '.join(_esc(o) for o in d['outputs'])
    target_pct  = _pct(d['q90_target'])
    ptr         = _pct(d.get('pct_train'))
    pvl         = _pct(d.get('pct_val'))
    pts         = _pct(d.get('pct_test'))
    best_esc    = _esc(best)

    # Performance table
    perf_rows = []
    for r in rows:
        status = (r'\cellcolor{passgreen}\textbf{PASS}'
                  if r['passed'] else r'\cellcolor{failred}\textbf{FAIL}')
        perf_rows.append(
            _esc(r['output']) + ' & '
            + _r2cell(r['r2']) + ' & '
            + _q90cell(r['q90'], r['passed']) + ' & '
            + target_pct + ' & '
            + _gapcell(r['gap']) + ' & '
            + status + r' \\ \hline'
        )
    perf_table = '\n  '.join(perf_rows)
    rec_items  = '\n  '.join(r'\item ' + b for b in recs)

    # Scatter (winner only)
    scatter_tex = ''
    path = scatter_paths.get(best, '')
    if path:
        scatter_tex = (
            rf'\exhead{{Predicted vs True --- {best_esc} (Test Set, up to 5\,000 points)}}' + '\n'
            r'\begin{center}' + '\n'
            rf'  \includegraphics[width=0.82\linewidth]{{{path}}}' + '\n'
            r'\end{center}'
        )

    # KS compact (winner only)
    ks_cells = []
    for o in d['outputs']:
        pval = d['dist'].get(best, {}).get(o)
        ks_cells.append(_esc(o) + ' & ' + _kscell(pval) + r' \\ \hline')
    ks_tex = '\n  '.join(ks_cells)

    return (
        '% ════ PART 1 — EXECUTIVE SUMMARY ════\n'
        r'\thispagestyle{fancy}' + '\n\n'
        r'\begin{center}' + '\n'
        rf'  {{\large\bfseries\color{{primary}} Executive Summary --- Surrogate Model Validation Report}}\\[2pt]' + '\n'
        rf'  {{\small\color{{sepgray}} Use case: \textbf{{{uc}}} \quad|\quad {date_str} \quad|\quad {n_models} model(s) evaluated}}' + '\n'
        r'\end{center}' + '\n'
        r'\vspace{-4pt}\noindent\rule{\linewidth}{1pt}\vspace{3pt}' + '\n\n'
        + _banner(d, best, rows) + '\n\n'
        r'\vspace{4pt}' + '\n\n'
        rf'\exhead{{Project Overview}}' + '\n'
        r'\renewcommand{\arraystretch}{1.2}' + '\n'
        r'\begin{tabular}{|l|p{0.60\linewidth}|}' + '\n'
        r'  \hline' + '\n'
        rf'  \textbf{{Use case}}           & {uc} \\ \hline' + '\n'
        rf'  \textbf{{Inputs}}             & {inputs_str} \\ \hline' + '\n'
        rf'  \textbf{{Outputs}}            & {n_out}: {outputs_str} \\ \hline' + '\n'
        rf'  \textbf{{Train / Val / Test}} & {ptr} / {pvl} / {pts} \\ \hline' + '\n'
        rf'  \textbf{{Accuracy target}}    & Q90 $<$ {target_pct} relative error per output \\ \hline' + '\n'
        r'\end{tabular}' + '\n\n'
        r'\vspace{4pt}' + '\n\n'
        rf'\exhead{{Model Performance --- \textbf{{{best_esc}}} (Test Set)}}' + '\n'
        r'\renewcommand{\arraystretch}{1.2}' + '\n'
        r'\begin{tabular}{|l|r|r|r|r|c|}' + '\n'
        r'  \hline' + '\n'
        r'  \textbf{Output} & \textbf{R\textsuperscript{2}} & \textbf{Q90} &'
        r' \textbf{Target} & \textbf{Gap} & \textbf{Status} \\ \hline' + '\n'
        f'  {perf_table}\n'
        r'\end{tabular}' + '\n'
        r'{\footnotesize' + '\n'
        rf'  \colorbox{{passgreen}}{{\ }} Q90 $<$ {target_pct}\enspace'
        rf'  \colorbox{{failred}}{{\ }} Q90 $\geq$ {target_pct}\enspace'
        r'  \colorbox{r2green}{\ } R\textsuperscript{2} $\geq 0.95$\enspace'
        r'  \colorbox{r2amber}{\ } R\textsuperscript{2} $\in[0.80,0.95)$\enspace'
        r'  Gap $=$ Q90$-$target ($<0$\,$\Rightarrow$\,margin)' + '\n'
        r'}' + '\n\n'
        r'\vspace{4pt}' + '\n\n'
        r'\exhead{Analysis \& Recommendations}' + '\n'
        r'\begin{itemize}[leftmargin=1.2em,itemsep=1pt,topsep=1pt]' + '\n'
        f'  {rec_items}\n'
        r'\end{itemize}' + '\n\n'
        r'\vspace{4pt}' + '\n\n'
        + scatter_tex + '\n\n'
        r'\vspace{4pt}' + '\n\n'
        rf'\exhead{{Overfitting Check --- KS Test p-values ({best_esc})}}' + '\n'
        r'{\footnotesize $H_0$: residuals on train and test follow the same distribution.\enspace'
        r'\colorbox{passgreen}{$p \geq 0.05$} no overfitting.\enspace'
        r'\colorbox{failred}{$p < 0.05$} overfitting.}\\[3pt]' + '\n'
        r'\renewcommand{\arraystretch}{1.15}' + '\n'
        r'\begin{tabular}{|l|r|}' + '\n'
        r'  \hline' + '\n'
        r'  \textbf{Output} & \textbf{KS p-value} \\ \hline' + '\n'
        f'  {ks_tex}\n'
        r'\end{tabular}' + '\n\n'
        r'\vspace{6pt}' + '\n'
        r'\noindent\textcolor{sepgray}{\small\textit{%' + '\n'
        r'  Full technical details in the'
        r' \hyperref[sec:techreview]{Technical Review} (page~\pageref{sec:techreview}).%' + '\n'
        r'}}' + '\n'
    )


# ── Part 2: Model Comparison ───────────────────────────────────────────────────

def _model_comparison(d):
    models     = d['models']
    outputs    = d['outputs']
    scores     = d['scores']
    vres       = d['vres']
    n_out      = len(outputs)
    n_mdl      = len(models)
    best       = d['best_model']
    target_pct = _pct(d['q90_target'])
    vres_map   = {vr['output']: vr for vr in vres}
    uc         = _esc(d['use_case'])
    date_str   = datetime.today().strftime('%d %B %Y')

    # Summary comparison table
    comp_rows = []
    for m in models:
        q90p = d['q90_pass'].get(m, 0)
        ksp  = d['ks_pass'].get(m, 0)
        avg  = d['avg_r2'].get(m, 0)
        b    = r'\bfseries' if m == best else ''
        star = r'\ $\star$' if m == best else ''
        comp_rows.append(
            rf'  {{{b} {_esc(m)}{star}}} & {{{b} {avg:.4f}}} '
            rf'& {{{b} {q90p}/{n_out}}} & {{{b} {ksp}/{n_out}}} \\ \hline'
        )
    comp_tex = '\n'.join(comp_rows)

    # Full Q90+R² per output per model
    col_spec = 'l' + '|rrr' * n_mdl
    _star = {m: (r'\ $\star$' if m == best else '') for m in models}
    mdl_span = ' & '.join(
        rf'\multicolumn{{3}}{{c|}}{{\textbf{{{_esc(m)}}}{_star[m]}}}'
        for m in models
    )
    sub_hdr  = (r' & R\textsuperscript{2} & Q90 & Gap') * n_mdl

    metric_rows = []
    for o in outputs:
        cells = [_esc(o)]
        for m in models:
            r2   = scores.get(m, {}).get(o, {}).get('R2')
            mres = vres_map.get(o, {}).get('models', {}).get(m, {})
            q90  = mres.get('score')
            ok   = mres.get('passed', False)
            gap  = round(q90 - d['q90_target'], 6) if q90 is not None else None
            cells += [_r2cell(r2), _q90cell(q90, ok), _gapcell(gap)]
        metric_rows.append(' & '.join(cells) + r' \\ \hline')
    metrics_tex = '\n  '.join(metric_rows)

    return (
        '% ════ PART 2 — MODEL COMPARISON ════\n'
        r'\clearpage' + '\n'
        r'\begin{center}' + '\n'
        rf'  {{\large\bfseries\color{{primary}} Model Comparison --- {uc}}}\\[2pt]' + '\n'
        rf'  {{\small\color{{sepgray}} {date_str}}}' + '\n'
        r'\end{center}' + '\n'
        r'\vspace{-4pt}\noindent\rule{\linewidth}{1pt}\vspace{6pt}' + '\n\n'
        r'\exhead{Overall Summary}' + '\n'
        r'\renewcommand{\arraystretch}{1.2}' + '\n'
        r'\begin{tabular}{|l|r|r|r|}' + '\n'
        r'  \hline' + '\n'
        r'  \textbf{Model} & \textbf{Avg R\textsuperscript{2}} &'
        rf' \textbf{{Q90 pass ({n_out} outputs)}} & \textbf{{KS pass ({n_out} outputs)}} \\ \hline' + '\n'
        + comp_tex + '\n'
        r'\end{tabular}' + '\n'
        r'{\footnotesize $\star$ Recommended model.\enspace'
        rf' Q90 target: $<{target_pct}$.\enspace KS: $p\geq 0.05$ = no overfitting.}}' + '\n\n'
        r'\vspace{6pt}' + '\n\n'
        r'\exhead{Per-Output Metrics (all models)}' + '\n'
        r'\renewcommand{\arraystretch}{1.2}' + '\n'
        rf'\begin{{tabular}}{{{col_spec}|}}' + '\n'
        r'  \hline' + '\n'
        rf'  \textbf{{Output}} & {mdl_span} \\ \hline' + '\n'
        rf'  & {sub_hdr[3:]} \\ \hline' + '\n'
        f'  {metrics_tex}\n'
        r'\end{tabular}' + '\n'
        r'{\footnotesize'
        r'  \colorbox{r2green}{\ } R\textsuperscript{2} $\geq 0.95$\enspace'
        r'  \colorbox{r2amber}{\ } R\textsuperscript{2} $\in[0.80,0.95)$\enspace'
        r'  \colorbox{failred}{\ } Fail\enspace'
        r'  Gap $=$ Q90$-$target}' + '\n'
    )


# ── Separator page ─────────────────────────────────────────────────────────────

def _separator(uc):
    return (
        r'\clearpage' + '\n'
        r'\thispagestyle{empty}' + '\n'
        r'\vspace*{\fill}' + '\n'
        r'\begin{center}' + '\n'
        r'  \textcolor{sepgray}{\rule{0.30\linewidth}{0.6pt}}\\[14pt]' + '\n'
        rf'  {{\Large\bfseries\color{{primary}} Technical Review}}\\[6pt]' + '\n'
        rf'  {{\normalsize\color{{sepgray}} {_esc(uc)} --- Full Validation Report}}\\[14pt]' + '\n'
        r'  \textcolor{sepgray}{\rule{0.30\linewidth}{0.6pt}}' + '\n'
        r'\end{center}' + '\n'
        r'\vspace*{\fill}' + '\n'
        r'\clearpage' + '\n'
    )


# ── Part 3: Technical Review ───────────────────────────────────────────────────

def _tech_review(d, scatter_paths, ratio_paths):
    models     = d['models']
    outputs    = d['outputs']
    scores     = d['scores']
    dist       = d['dist']
    vres       = d['vres']
    split      = d['split']
    best       = d['best_model']
    n_out      = len(outputs)
    n_mdl      = len(models)
    target_pct = _pct(d['q90_target'])
    vres_map   = {vr['output']: vr for vr in vres}
    uc_esc     = _esc(d['use_case'])
    best_esc   = _esc(best)
    best_r2    = d['avg_r2'].get(best, 0)
    best_q90p  = d['q90_pass'].get(best, 0)
    best_ksp   = d['ks_pass'].get(best, 0)
    model_list = ', '.join(_esc(m) for m in models)
    inputs_str = ', '.join(_esc(i) for i in d['inputs'])
    n_inputs   = len(d['inputs'])

    # ── Introduction prose ────────────────────────────────────────────────────
    intro = (
        rf'This report documents the full validation of surrogate models for use case '
        rf'\textbf{{{uc_esc}}}. {n_mdl} model architecture(s) were trained and evaluated: '
        rf'{model_list}. Each model predicts {n_out} structural output quantities from '
        rf'{n_inputs} input features ({inputs_str}). '
        rf'The accuracy criterion requires the 90th-percentile relative error (Q90) '
        rf'to remain below {target_pct} for every output on unseen test data.'
    )

    # ── Model selection prose ─────────────────────────────────────────────────
    runners = [m for m in models if m != best]
    if runners:
        ru_str = ', '.join(
            rf'{_esc(m)} (R\textsuperscript{{2}}={d["avg_r2"].get(m,0):.4f})'
            for m in runners
        )
        sel_prose = (
            rf'\textbf{{{best_esc}}} achieved the highest average R\textsuperscript{{2}} '
            rf'on the test set ({best_r2:.4f}), outperforming {ru_str}. '
            rf'It is the recommended model for deployment.'
        )
    else:
        sel_prose = (
            rf'\textbf{{{best_esc}}} is the sole model evaluated '
            rf'(avg R\textsuperscript{{2}} = {best_r2:.4f}).'
        )

    # ── Full metrics table (all models) ───────────────────────────────────────
    col_spec  = 'l' + '|rrr' * n_mdl
    _star2    = {m: (r'\ $\star$' if m == best else '') for m in models}
    mdl_span  = ' & '.join(
        rf'\multicolumn{{3}}{{c|}}{{\textbf{{{_esc(m)}}}{_star2[m]}}}'
        for m in models
    )
    sub_hdr   = (r' & R\textsuperscript{2} & Q90 & Gap') * n_mdl
    mrows = []
    for o in outputs:
        cells = [_esc(o)]
        for m in models:
            r2   = scores.get(m, {}).get(o, {}).get('R2')
            mres = vres_map.get(o, {}).get('models', {}).get(m, {})
            q90  = mres.get('score')
            ok   = mres.get('passed', False)
            gap  = round(q90 - d['q90_target'], 6) if q90 is not None else None
            cells += [_r2cell(r2), _q90cell(q90, ok), _gapcell(gap)]
        mrows.append(' & '.join(cells) + r' \\ \hline')
    metrics_tex = '\n  '.join(mrows)

    # Accuracy prose
    fail_names = [r['output'] for r in _per_output(d, best) if not r['passed']]
    pass_names = [r['output'] for r in _per_output(d, best) if r['passed']]
    if best_q90p == n_out:
        acc_prose = (
            rf'\textbf{{{best_esc}}} satisfies the Q90 requirement on '
            rf'all {n_out} outputs. No corrective action is required for deployment.'
        )
    else:
        fn = ', '.join(_esc(o) for o in fail_names)
        pn = ', '.join(_esc(o) for o in pass_names)
        acc_prose = (
            rf'\textbf{{{best_esc}}} satisfies Q90 on {best_q90p}/{n_out} outputs. '
            rf'Passing: {pn}. '
            rf'Failing: \textcolor{{red}}{{\textit{{{fn}}}}}. '
            rf'Refer to the Improvement Roadmap for targeted corrective actions.'
        )

    # ── Scatter plots (all models) ────────────────────────────────────────────
    scatter_tex = ''
    for m in models:
        path = scatter_paths.get(m, '')
        if path:
            m_esc = _esc(m)
            label = rf'{m_esc}' + (r' $\star$' if m == best else '')
            scatter_tex += (
                rf'\subsection*{{\normalsize {label}}}' + '\n'
                r'\begin{center}' + '\n'
                rf'  \includegraphics[width=0.85\linewidth]{{{path}}}' + '\n'
                r'\end{center}' + '\n\n'
            )

    # ── Ratio / relative-error plots (all models) ─────────────────────────────
    ratio_tex = ''
    for m in models:
        path = ratio_paths.get(m, '')
        if path:
            m_esc = _esc(m)
            label = rf'{m_esc}' + (r' $\star$' if m == best else '')
            ratio_tex += (
                rf'\subsection*{{\normalsize {label}}}' + '\n'
                r'\begin{center}' + '\n'
                rf'  \includegraphics[width=0.85\linewidth]{{{path}}}' + '\n'
                r'\end{center}' + '\n\n'
            )

    # ── KS test (all models) ──────────────────────────────────────────────────
    ks_col  = '|l' + '|r' * n_mdl + '|'
    ks_hdrs = ' & '.join(r'\textbf{' + _esc(m) + '}' for m in models)
    ks_rows = []
    for o in outputs:
        cells = [_esc(o)]
        for m in models:
            cells.append(_kscell(d['dist'].get(m, {}).get(o)))
        ks_rows.append(' & '.join(cells) + r' \\ \hline')
    ks_tex = '\n  '.join(ks_rows)

    # KS prose
    ks_fail_names = [o for o in outputs
                     if (dist.get(best, {}).get(o) or 1.0) < 0.05]
    if best_ksp == n_out:
        ks_prose = (
            rf'The KS test confirms statistically consistent residual distributions '
            rf'between train and test sets for all {n_out} outputs of '
            rf'\textbf{{{best_esc}}} ($p \geq 0.05$ in every case). '
            rf'There is no evidence of overfitting.'
        )
    else:
        kfn = ', '.join(_esc(o) for o in ks_fail_names)
        ks_prose = (
            rf'The KS test detects distributional differences for '
            rf'\textbf{{{best_esc}}} on: \textcolor{{red}}{{\textit{{{kfn}}}}}. '
            rf'This suggests localised overfitting on these outputs. '
            rf'Increasing regularisation or reducing model capacity is recommended.'
        )

    # ── Split quality ─────────────────────────────────────────────────────────
    def _sv(key, fmt='.3f'):
        v = split.get(key)
        return f'{v:{fmt}}' if v is not None else r'\textemdash'

    def _sicon(key, op, thresh):
        v = split.get(key)
        if v is None:
            return r'\textemdash'
        ok = (v <= thresh) if op == '<=' else (v >= thresh)
        return (r'\cellcolor{passgreen}$\checkmark$' if ok
                else r'\cellcolor{failred}$\times$')

    rvp = split.get('residual_voxel_proportion', 1.0)
    chi = split.get('chi_squared_pvalue') or 0.0
    split_prose = (
        r'The VTP analysis confirms the test set is well-distributed within '
        r'the training space (residual voxel proportion $\leq 0.05$, no isolated '
        r'test points). The chi-squared test validates the split is statistically '
        r'unbiased. Train/test partition is of high quality.'
        if rvp <= 0.05 and chi >= 0.05 else
        r'One or more split quality metrics fall outside acceptable bounds. '
        r'Review the data collection strategy and consider resampling the split '
        r'or collecting additional data in sparse input regions.'
    )

    # ── Detailed recommendations ──────────────────────────────────────────────
    detail_items = '\n  '.join(
        r'\item ' + b
        for b in _recommendations(d, best, _per_output(d, best), detailed=True)
    )

    return (
        '% ════ PART 3 — TECHNICAL REVIEW ════\n'
        r'\label{sec:techreview}' + '\n'
        r'\phantomsection' + '\n'
        r'\addcontentsline{toc}{section}{Technical Review}' + '\n\n'
        r'\begin{center}' + '\n'
        rf'  {{\normalsize\bfseries\color{{primary}} Technical Review --- {uc_esc}}}' + '\n'
        r'\end{center}' + '\n'
        r'\tableofcontents' + '\n'
        r'\clearpage' + '\n\n'

        r'\section{Introduction}' + '\n'
        r'\label{sec:intro}' + '\n\n'
        + intro + '\n\n'

        r'\section{Model Selection}' + '\n'
        r'\label{sec:selection}' + '\n\n'
        + sel_prose + '\n\n'

        r'\section{Accuracy Assessment (Q90 \& R\textsuperscript{2})}' + '\n'
        r'\label{sec:accuracy}' + '\n\n'
        + acc_prose + '\n\n'
        r'\vspace{4pt}' + '\n'
        r'\renewcommand{\arraystretch}{1.2}' + '\n'
        rf'\begin{{tabular}}{{{col_spec}|}}' + '\n'
        r'  \hline' + '\n'
        rf'  \textbf{{Output}} & {mdl_span} \\ \hline' + '\n'
        rf'  & {sub_hdr[3:]} \\ \hline' + '\n'
        f'  {metrics_tex}\n'
        r'\end{tabular}' + '\n'
        r'{\footnotesize'
        r'  \colorbox{r2green}{\ } R\textsuperscript{2} $\geq 0.95$\enspace'
        r'  \colorbox{r2amber}{\ } R\textsuperscript{2} $\in[0.80,0.95)$\enspace'
        r'  \colorbox{failred}{\ } Fail\enspace'
        r'  Gap $=$ Q90$-$target ($<0$\,$\Rightarrow$\,margin)}' + '\n\n'

        r'\section{Predicted vs True (Scatter Plots, Test Set)}' + '\n'
        r'\label{sec:scatter}' + '\n\n'
        r'Each plot shows up to 5\,000 test-set points. '
        r'The dashed diagonal represents perfect prediction ($\hat{y}=y$). '
        r'Systematic bias or fan-shaped spread (heteroscedasticity) indicate model limitations.' + '\n\n'
        + scatter_tex

        + (
            r'\section{Relative Error Analysis (Ratio Plots, Test Set)}' + '\n'
            r'\label{sec:ratio}' + '\n\n'
            r'Each panel shows the ratio $\hat{y}/y$ as a function of the true value. '
            r'A ratio of 1.0 (dashed line) corresponds to perfect prediction. '
            r'Systematic deviation from 1.0 reveals bias; increasing spread with magnitude '
            r'reveals heteroscedasticity.' + '\n\n'
            + ratio_tex
            if ratio_tex else ''
        )

        + r'\section{Overfitting Assessment (KS Test)}' + '\n'
        r'\label{sec:ks}' + '\n\n'
        r'The Kolmogorov--Smirnov test compares residual distributions on train vs test sets. '
        r'$H_0$: both distributions are identical (no overfitting). '
        r'$p < 0.05$ is statistical evidence of overfitting.' + '\n\n'
        + ks_prose + '\n\n'
        r'\vspace{4pt}' + '\n'
        r'\renewcommand{\arraystretch}{1.2}' + '\n'
        rf'\begin{{tabular}}{{{ks_col}}}' + '\n'
        r'  \hline' + '\n'
        rf'  \textbf{{Output}} & {ks_hdrs} \\ \hline' + '\n'
        f'  {ks_tex}\n'
        r'\end{tabular}' + '\n'
        r'{\footnotesize'
        r'  \colorbox{passgreen}{\ } $p \geq 0.05$ no overfitting\enspace'
        r'  \colorbox{failred}{\ } $p < 0.05$ overfitting detected}' + '\n\n'

        + r'\section{Data Split Quality (VTP Analysis)}' + '\n'
        r'\label{sec:split}' + '\n\n'
        r'The Voronoi Tesselation Proximity method verifies whether test points '
        r'lie within the training distribution. '
        r'A residual voxel proportion $\leq 0.05$ and chi-squared $p \geq 0.05$ '
        r'confirm a high-quality, unbiased split.' + '\n\n'
        + split_prose + '\n\n'
        r'\vspace{4pt}' + '\n'
        r'\renewcommand{\arraystretch}{1.2}' + '\n'
        r'\begin{tabular}{|l|r|c|}' + '\n'
        r'  \hline' + '\n'
        r'  \textbf{Metric} & \textbf{Value} & \textbf{Pass} \\ \hline' + '\n'
        rf'  Residual voxel proportion ($\leq 0.05$) & {_sv("residual_voxel_proportion")} & {_sicon("residual_voxel_proportion","<=",0.05)} \\ \hline' + '\n'
        rf'  Valid test proportion & {_sv("valid_test_proportion")} & \\ \hline' + '\n'
        rf'  Phacking test proportion & {_sv("phacking_test_proportion")} & \\ \hline' + '\n'
        rf'  Isolated test proportion & {_sv("isolated_test_proportion")} & \\ \hline' + '\n'
        rf'  Chi\textsuperscript{{2}} p-value ($\geq 0.05$) & {_sv("chi_squared_pvalue",".4f")} & {_sicon("chi_squared_pvalue",">=",0.05)} \\ \hline' + '\n'
        r'\end{tabular}' + '\n\n'

        + r'\section{Improvement Roadmap}' + '\n'
        r'\label{sec:roadmap}' + '\n\n'
        r'Prioritised actions based on the failure analysis above.' + '\n\n'
        r'\begin{itemize}[leftmargin=1.4em,itemsep=3pt,topsep=3pt]' + '\n'
        f'  {detail_items}\n'
        r'\end{itemize}' + '\n'
    )


# ── Main builder ───────────────────────────────────────────────────────────────

def build_latex(d: dict, scatter_paths: dict, ratio_paths: dict) -> str:
    best = d['best_model']
    rows = _per_output(d, best)
    recs = _recommendations(d, best, rows, detailed=False)
    uc   = d['use_case']

    preamble = _PREAMBLE.replace('XUSECASEX', _esc(uc))

    return (
        preamble
        + r'\begin{document}' + '\n'
        + _part1(d, best, rows, recs, scatter_paths)
        + _model_comparison(d)
        + _separator(uc)
        + _tech_review(d, scatter_paths, ratio_paths)
        + '\n' + r'\end{document}' + '\n'
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate a professional LaTeX/PDF report from SF metadata JSON.')
    parser.add_argument('metadata', help='Path to metadata_*.json')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: same folder as metadata)')
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    out_dir   = (Path(args.output).resolve()
                 if args.output else meta_path.parent.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)

    d = extract(meta_path)

    artifacts_dir = meta_path.parent.resolve()
    scatter_paths, ratio_paths = {}, {}
    for m in d['models']:
        for kind, store in [('scatter', scatter_paths), ('ratio', ratio_paths)]:
            png = artifacts_dir / f'{kind}_{m}.png'
            if png.exists():
                store[m] = os.path.relpath(str(png), str(out_dir))

    tex = build_latex(d, scatter_paths, ratio_paths)

    use_case = d['use_case']
    tex_file = out_dir / f'executive_summary_{use_case}.tex'
    pdf_file = out_dir / f'executive_summary_{use_case}.pdf'

    tex_file.write_text(tex, encoding='utf-8')
    print(f'LaTeX source → {tex_file}')

    result = subprocess.run(
        ['tectonic', '-o', str(out_dir), str(tex_file)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f'PDF generated  → {pdf_file}')
    else:
        print('tectonic STDERR:\n', result.stderr[-3000:])
        print('LaTeX source saved — compile manually.')
        sys.exit(1)


if __name__ == '__main__':
    main()
