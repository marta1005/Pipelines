"""
Generate a professional two-part PDF validation report from a Surrogate Factory
metadata JSON.

Structure
---------
  Part 1 (≤ 2 pages)  Executive Summary — winner model, verdict banner with
                       colour-coded metric rows, analysis, scatter plot, KS check.
  Separator page       Decorative divider.
  Part 2               Technical Appendix — full prose report mirroring the
                       validation template, with TOC on its own page.

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
    """Return a LaTeX-safe percentage string, e.g. 0.10 → '10\\%'."""
    if v is None:
        return r'\textemdash'
    fmt = f'{v * 100:.{decimals}f}'
    return rf'{fmt}\%'


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


# ── Per-output stats ───────────────────────────────────────────────────────────

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
    """Return list of LaTeX bullet strings. detailed=True adds more context."""
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
            extra = (
                r' Collect additional flight-condition data near the boundaries '
                r'where these outputs show the largest residuals.'
                if detailed else ''
            )
            bullets.append(
                rf'\textbf{{Marginal failures}} (\textit{{{names}}}): Q90 is within '
                rf'5\% of target. Targeted data augmentation is likely sufficient.{extra}'
            )
        if severe:
            names = ', '.join(_esc(r['output']) for r in severe)
            extra = (
                r' Consider whether additional physical drivers of these outputs '
                r'are missing from the input feature set, or evaluate a more '
                r'expressive architecture (e.g.\ deeper network, ensemble).'
                if detailed else ''
            )
            bullets.append(
                rf'\textbf{{Significant failures}} (\textit{{{names}}}): '
                rf'Q90 exceeds target by $\geq 5\%$. Investigate input features '
                rf'and model capacity.{extra}'
            )

    if low_r2:
        names = ', '.join(_esc(r['output']) for r in low_r2)
        extra = (
            r' A low R\textsuperscript{2} often indicates missing physical '
            r'drivers or highly nonlinear behaviour that requires richer features '
            r'or a more flexible model.'
            if detailed else ''
        )
        bullets.append(
            rf'\textbf{{Low R\textsuperscript{{2}}}} (\textit{{{names}}}): '
            rf'Coefficient of determination below 0.80 --- high unexplained variance.{extra}'
        )

    if ks_fails:
        names = ', '.join(_esc(o) for o in ks_fails)
        extra = (
            r' Increase regularisation strength (dropout rate, weight decay, '
            r'early-stopping patience) or reduce model capacity.'
            if detailed else ''
        )
        bullets.append(
            rf'\textbf{{Overfitting detected}} (\textit{{{names}}}): KS test '
            rf'rejects equal residual distributions on train and test.{extra}'
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

% ── Section style (Part 2 only) ───────────────────────────────────────────────
\titleformat{\section}{\normalsize\bfseries\color{primary}}{}{0em}{}
            [\vspace{-4pt}\textcolor{primary}{\rule{\linewidth}{0.5pt}}]
\titlespacing{\section}{0pt}{10pt}{4pt}

% ── Lightweight heading safe inside minipage / Part 1 ─────────────────────────
\newcommand{\exhead}[1]{%
  \medskip\noindent{\small\bfseries\color{primary}#1}%
  \par\vspace{-5pt}\noindent\textcolor{primary}{\rule{\linewidth}{0.4pt}}\vspace{2pt}}

% ── Page header / footer ──────────────────────────────────────────────────────
\pagestyle{fancy}\fancyhf{}
\lhead{\small\textcolor{primary}{\textbf{Surrogate Factory}}}
\chead{\small\textcolor{sepgray}{CONFIDENTIAL}}
\rhead{\small\textcolor{primary}{XUSECASEX}}
\lfoot{\small\textcolor{sepgray}{Surrogate Factory v2.2 --- Automated Validation Report}}
\rfoot{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}
"""


# ── Part 1: Executive Summary ──────────────────────────────────────────────────

def _banner(d, best):
    """Colour-coded verdict banner with one row per metric."""
    n_out     = len(d['outputs'])
    best_r2   = d['avg_r2'].get(best, 0)
    best_q90p = d['q90_pass'].get(best, 0)
    best_ksp  = d['ks_pass'].get(best, 0)

    overall_ok  = (best_q90p == n_out) and (best_ksp == n_out)
    box_color   = 'passgreen' if overall_ok else 'warnamber'
    verdict_txt = 'ALL REQUIREMENTS MET' if overall_ok else 'REQUIREMENTS PARTIALLY MET'

    # R² metric row
    if best_r2 >= 0.95:
        r2_color, r2_label = 'r2green',  'Excellent'
    elif best_r2 >= 0.80:
        r2_color, r2_label = 'r2amber',  'Acceptable'
    else:
        r2_color, r2_label = 'failred',  'Poor --- needs attention'

    # Q90 metric row
    if best_q90p == n_out:
        q90_color, q90_label = 'passgreen', 'All outputs meet requirements'
    elif best_q90p > 0:
        q90_color, q90_label = 'warnamber', f'{best_q90p}/{n_out} outputs pass --- partially met'
    else:
        q90_color, q90_label = 'failred',   'No output meets requirements'

    # KS metric row
    if best_ksp == n_out:
        ks_color, ks_label = 'passgreen', 'No overfitting detected'
    elif best_ksp > n_out // 2:
        ks_color, ks_label = 'warnamber', f'{best_ksp}/{n_out} outputs pass --- minor overfitting'
    else:
        ks_color, ks_label = 'failred',   f'{best_ksp}/{n_out} outputs pass --- overfitting present'

    best_esc  = _esc(best)
    r2_str    = f'{best_r2:.4f}'
    q90p_str  = str(best_q90p)
    ksp_str   = str(best_ksp)
    nout_str  = str(n_out)

    return rf"""\begin{{center}}
\colorbox{{{box_color}}}{{\begin{{minipage}}{{0.62\linewidth}}
  \vspace{{5pt}}\centering
  {{\normalsize\bfseries {verdict_txt}}}\\[5pt]
  {{\small\bfseries Recommended model:}} {{\small {best_esc}}}\\[5pt]
  \colorbox{{{r2_color}}}{{\begin{{minipage}}{{0.97\linewidth}}
    \vspace{{2pt}}\centering
    {{\small\textbf{{Avg R\textsuperscript{{2}}: {r2_str}}} \enspace---\enspace {r2_label}}}
    \vspace{{2pt}}
  \end{{minipage}}}}\\[2pt]
  \colorbox{{{q90_color}}}{{\begin{{minipage}}{{0.97\linewidth}}
    \vspace{{2pt}}\centering
    {{\small\textbf{{Q90 accuracy: {q90p_str}/{nout_str} outputs}} \enspace---\enspace {q90_label}}}
    \vspace{{2pt}}
  \end{{minipage}}}}\\[2pt]
  \colorbox{{{ks_color}}}{{\begin{{minipage}}{{0.97\linewidth}}
    \vspace{{2pt}}\centering
    {{\small\textbf{{Overfitting check: {ksp_str}/{nout_str} outputs}} \enspace---\enspace {ks_label}}}
    \vspace{{2pt}}
  \end{{minipage}}}}
  \vspace{{5pt}}
\end{{minipage}}}}
\end{{center}}"""


def _part1(d, best, rows, recs, scatter_paths):
    uc         = _esc(d['use_case'])
    n_out      = len(d['outputs'])
    n_models   = len(d['models'])
    date_str   = datetime.today().strftime('%d %B %Y')
    inputs_str = ', '.join(_esc(i) for i in d['inputs'])
    outputs_str= ', '.join(_esc(o) for o in d['outputs'])
    target_pct = _pct(d['q90_target'])
    ptr        = _pct(d.get('pct_train'))
    pvl        = _pct(d.get('pct_val'))
    pts        = _pct(d.get('pct_test'))

    # ── Performance table (winner only, with \hline) ───────────────────────────
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

    # ── Recommendations ────────────────────────────────────────────────────────
    rec_items = '\n  '.join(r'\item ' + b for b in recs)

    # ── Scatter plot (winner only) ─────────────────────────────────────────────
    scatter_path = scatter_paths.get(best, '')
    scatter_tex = ''
    if scatter_path:
        best_esc = _esc(best)
        scatter_tex = rf"""\exhead{{Predicted vs True --- {best_esc} (Test Set, sample of 5\,000 points)}}
\begin{{center}}
  \includegraphics[width=0.82\linewidth]{{{scatter_path}}}
\end{{center}}"""

    # ── KS test compact (winner only) ─────────────────────────────────────────
    ks_cells = []
    for o in d['outputs']:
        pval = d['dist'].get(best, {}).get(o)
        ks_cells.append(_esc(o) + ' & ' + _kscell(pval) + r' \\ \hline')
    ks_tex = '\n  '.join(ks_cells)
    best_esc = _esc(best)

    return rf"""% ════════════════════════════════════════════════════════════════
%  PART 1 — EXECUTIVE SUMMARY
% ════════════════════════════════════════════════════════════════
\thispagestyle{{fancy}}

\begin{{center}}
  {{\large\bfseries\color{{primary}}
    Executive Summary --- Surrogate Model Validation Report}}\\[2pt]
  {{\small\color{{sepgray}}
    Use case: \textbf{{{uc}}}
    \quad|\quad {date_str}
    \quad|\quad {n_models} model(s) evaluated}}
\end{{center}}
\vspace{{-4pt}}\noindent\rule{{\linewidth}}{{1pt}}\vspace{{3pt}}

{_banner(d, best)}

\vspace{{4pt}}

\exhead{{Project Overview}}
\renewcommand{{\arraystretch}}{{1.2}}
\begin{{tabular}}{{|l|p{{0.60\linewidth}}|}}
  \hline
  \textbf{{Use case}}          & {uc} \\ \hline
  \textbf{{Inputs}}            & {inputs_str} \\ \hline
  \textbf{{Outputs}}           & {n_out}: {outputs_str} \\ \hline
  \textbf{{Train / Val / Test}}& {ptr} / {pvl} / {pts} \\ \hline
  \textbf{{Accuracy target}}   & Q90 $<$ {target_pct} relative error per output \\ \hline
\end{{tabular}}

\vspace{{4pt}}

\exhead{{Model Performance --- \textbf{{{best_esc}}} (Test Set)}}
\renewcommand{{\arraystretch}}{{1.2}}
\begin{{tabular}}{{|l|r|r|r|r|c|}}
  \hline
  \textbf{{Output}} &
  \textbf{{R\textsuperscript{{2}}}} &
  \textbf{{Q90}} &
  \textbf{{Target}} &
  \textbf{{Gap}} &
  \textbf{{Status}} \\ \hline
  {perf_table}
\end{{tabular}}

\smallskip
{{\footnotesize
  \colorbox{{passgreen}}{{\ }} Q90 $<$ {target_pct}\enspace
  \colorbox{{failred}}{{\ }} Q90 $\geq$ {target_pct}\enspace
  \colorbox{{r2green}}{{\ }} R\textsuperscript{{2}} $\geq 0.95$\enspace
  \colorbox{{r2amber}}{{\ }} R\textsuperscript{{2}} $\in[0.80,0.95)$\enspace
  Gap $=$ Q90\,$-$\,target (negative\,$\Rightarrow$\,margin)
}}

\vspace{{4pt}}

\exhead{{Analysis \& Recommendations}}
\begin{{itemize}}[leftmargin=1.2em,itemsep=1pt,topsep=1pt]
  {rec_items}
\end{{itemize}}

\vspace{{4pt}}

{scatter_tex}

\vspace{{4pt}}

\exhead{{Overfitting Check --- KS Test p-values ({best_esc})}}
{{\footnotesize $H_0$: residuals on train and test follow the same distribution.
\colorbox{{passgreen}}{{$p \geq 0.05$}} = no overfitting.\enspace
\colorbox{{failred}}{{$p < 0.05$}} = overfitting.}}\\[3pt]
\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabular}}{{|l|r|}}
  \hline
  \textbf{{Output}} & \textbf{{KS p-value}} \\ \hline
  {ks_tex}
\end{{tabular}}

\vspace{{6pt}}
\noindent\textcolor{{sepgray}}{{\small\textit{{%
  Full technical details in the
  \hyperref[sec:appendix]{{Technical Appendix}} (page~\pageref{{sec:appendix}}).%
}}}}
"""


# ── Separator page ─────────────────────────────────────────────────────────────

def _separator(uc):
    return rf"""\clearpage
\thispagestyle{{empty}}
\vspace*{{\fill}}
\begin{{center}}
  \textcolor{{sepgray}}{{\rule{{0.30\linewidth}}{{0.6pt}}}}\\[14pt]
  {{\Large\bfseries\color{{primary}} Technical Appendix}}\\[6pt]
  {{\normalsize\color{{sepgray}} {_esc(uc)} --- Full Validation Report}}\\[14pt]
  \textcolor{{sepgray}}{{\rule{{0.30\linewidth}}{{0.6pt}}}}
\end{{center}}
\vspace*{{\fill}}
\clearpage
"""


# ── Part 2: Technical Appendix ─────────────────────────────────────────────────

def _part2(d, scatter_paths):
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

    # ── 0. Introduction prose ─────────────────────────────────────────────────
    model_list = ', '.join(_esc(m) for m in models)
    n_inputs   = len(d['inputs'])
    inputs_str = ', '.join(_esc(i) for i in d['inputs'])
    outputs_str= ', '.join(_esc(o) for o in d['outputs'])
    best_esc   = _esc(best)
    best_r2    = d['avg_r2'].get(best, 0)
    best_q90p  = d['q90_pass'].get(best, 0)
    best_ksp   = d['ks_pass'].get(best, 0)
    uc_esc     = _esc(d['use_case'])

    intro_prose = (
        rf'This appendix documents the full validation of surrogate models for use case '
        rf'\textbf{{{uc_esc}}}. A total of {n_mdl} model architecture(s) '
        rf'were trained and evaluated: {model_list}. '
        rf'The models predict {n_out} output quantities '
        rf'({outputs_str}) '
        rf'from {n_inputs} input features ({inputs_str}). '
        rf'The accuracy criterion requires that the 90th-percentile relative error (Q90) '
        rf'remains below {target_pct} for every output.'
    )

    # ── 1. Model comparison table ──────────────────────────────────────────────
    comp_rows = []
    for m in models:
        q90p = d['q90_pass'].get(m, 0)
        ksp  = d['ks_pass'].get(m, 0)
        avg  = d['avg_r2'].get(m, 0)
        is_best = (m == best)
        b = r'\bfseries' if is_best else ''
        star = r'\ $\star$' if is_best else ''
        comp_rows.append(
            rf'  {{{b} {_esc(m)}{star}}} & {{{b} {avg:.4f}}} '
            rf'& {{{b} {q90p}/{n_out}}} & {{{b} {ksp}/{n_out}}} \\ \hline'
        )
    comp_tex = '\n'.join(comp_rows)

    # Prose: rationale for winner
    runner_up = [m for m in models if m != best]
    if runner_up:
        ru_r2s = ', '.join(f'{_esc(m)} (R\\textsuperscript{{2}}={d["avg_r2"].get(m,0):.4f})'
                           for m in runner_up)
        selection_prose = (
            rf'\textbf{{{best_esc}}} achieved the highest average '
            rf'R\textsuperscript{{2}} on the test set ({best_r2:.4f}), '
            rf'outperforming {ru_r2s}. It is therefore the recommended model '
            rf'for production deployment.'
        )
    else:
        selection_prose = (
            rf'\textbf{{{best_esc}}} is the sole model evaluated '
            rf'(avg R\textsuperscript{{2}} = {best_r2:.4f}).'
        )

    # ── 2. Full Q90 / R² metrics table (all models) ───────────────────────────
    col_spec  = 'l' + '|rrr' * n_mdl
    mdl_span  = ' & '.join(
        rf'\multicolumn{{3}}{{c|}}{{\textbf{{{_esc(m)}}}}}'
        for m in models
    )
    sub_hdr   = (r' & R\textsuperscript{2} & Q90 & Gap') * n_mdl

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

    # Prose: accuracy summary
    all_pass = best_q90p == n_out
    fail_names = [r['output'] for r in _per_output(d, best) if not r['passed']]
    pass_names = [r['output'] for r in _per_output(d, best) if r['passed']]
    if all_pass:
        acc_prose = (
            rf'The recommended model \textbf{{{best_esc}}} satisfies the Q90 '
            rf'accuracy requirement on \textbf{{all {n_out} outputs}}. '
            rf'No corrective action is required for deployment.'
        )
    else:
        fn = ', '.join(_esc(o) for o in fail_names)
        pn = ', '.join(_esc(o) for o in pass_names)
        acc_prose = (
            rf'The recommended model \textbf{{{best_esc}}} satisfies Q90 on '
            rf'{best_q90p} out of {n_out} outputs. '
            rf'Passing outputs: {pn}. '
            rf'Failing outputs: \textcolor{{red}}{{{fn}}}. '
            rf'See the Analysis \& Recommendations section in the Executive Summary '
            rf'for targeted improvement actions.'
        )

    # ── 3. Scatter plots (all models) ─────────────────────────────────────────
    scatter_tex = ''
    for m in models:
        path = scatter_paths.get(m, '')
        if path:
            m_esc = _esc(m)
            scatter_tex += (
                rf'\subsection*{{\normalsize {m_esc}}}' + '\n'
                r'\begin{center}' + '\n'
                rf'  \includegraphics[width=0.82\linewidth]{{{path}}}' + '\n'
                r'\end{center}' + '\n\n'
            )

    # ── 4. KS test table (all models) ─────────────────────────────────────────
    ks_col   = '|l' + '|r' * n_mdl + '|'
    ks_hdrs  = ' & '.join(r'\textbf{' + _esc(m) + '}' for m in models)
    ks_rows2 = []
    for o in outputs:
        cells = [_esc(o)]
        for m in models:
            cells.append(_kscell(d['dist'].get(m, {}).get(o)))
        ks_rows2.append(' & '.join(cells) + r' \\ \hline')
    ks_tex2 = '\n  '.join(ks_rows2)

    # Prose: overfitting summary
    ks_ok = best_ksp == n_out
    if ks_ok:
        ks_prose = (
            rf'The KS test confirms that the residual distributions of '
            rf'\textbf{{{best_esc}}} are statistically consistent between the '
            rf'training and test sets for all {n_out} outputs ($p \geq 0.05$ in every case). '
            rf'There is no statistical evidence of overfitting.'
        )
    else:
        ks_fail_names = [o for o in outputs
                         if (d['dist'].get(best, {}).get(o) or 1.0) < 0.05]
        kfn = ', '.join(_esc(o) for o in ks_fail_names)
        ks_prose = (
            rf'The KS test detects distributional differences between train and test '
            rf'residuals for \textbf{{{best_esc}}} on the following outputs: '
            rf'\textcolor{{red}}{{\textit{{{kfn}}}}}. '
            rf'This suggests localised overfitting. '
            rf'Increasing regularisation or reducing model capacity for these outputs '
            rf'is recommended before deployment.'
        )

    # ── 5. Split quality ──────────────────────────────────────────────────────
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

    rvp_val  = _sv('residual_voxel_proportion')
    rvp_icon = _sicon('residual_voxel_proportion', '<=', 0.05)
    vtp_val  = _sv('valid_test_proportion')
    pht_val  = _sv('phacking_test_proportion')
    itp_val  = _sv('isolated_test_proportion')
    chi_val  = _sv('chi_squared_pvalue', '.4f')
    chi_icon = _sicon('chi_squared_pvalue', '>=', 0.05)

    rvp_num = split.get('residual_voxel_proportion', 1.0)
    chi_num = split.get('chi_squared_pvalue', 0.0) or 0.0
    if rvp_num <= 0.05 and chi_num >= 0.05:
        split_prose = (
            r'The Voronoi Tesselation Proximity (VTP) analysis confirms that the '
            r'test set is well-distributed within the training space '
            r'(residual voxel proportion $\leq 0.05$, no isolated test points). '
            r'The chi-squared test ($p \geq 0.05$) validates that the split is '
            r'statistically unbiased. The train/test partition is of high quality.'
        )
    else:
        split_prose = (
            r'One or more split quality metrics fall outside acceptable bounds. '
            r'The test set may contain points poorly covered by the training distribution. '
            r'Review the data collection strategy and consider resampling the split '
            r'or collecting additional data in sparse regions.'
        )

    # ── 6. Detailed recommendations ───────────────────────────────────────────
    detail_recs = _recommendations(d, best, _per_output(d, best), detailed=True)
    detail_items = '\n  '.join(r'\item ' + b for b in detail_recs)

    return rf"""% ════════════════════════════════════════════════════════════════
%  PART 2 — TECHNICAL APPENDIX
% ════════════════════════════════════════════════════════════════
\label{{sec:appendix}}
\phantomsection
\addcontentsline{{toc}}{{section}}{{Technical Appendix}}

\begin{{center}}
  {{\normalsize\bfseries\color{{primary}}
    Technical Appendix --- Full Validation Report --- {uc_esc}}}
\end{{center}}
\vspace{{2pt}}

\tableofcontents
\clearpage

% ── Introduction ──────────────────────────────────────────────────────────────
\section{{Introduction}}
\label{{sec:intro}}

{intro_prose}

% ── 1. Model Selection ────────────────────────────────────────────────────────
\section{{Model Selection}}
\label{{sec:selection}}

{selection_prose}

\vspace{{4pt}}
\renewcommand{{\arraystretch}}{{1.2}}
\begin{{tabular}}{{|l|r|r|r|}}
  \hline
  \textbf{{Model}} &
  \textbf{{Avg R\textsuperscript{{2}}}} &
  \textbf{{Q90 pass ({n_out} outputs)}} &
  \textbf{{KS pass ({n_out} outputs)}} \\ \hline
{comp_tex}
\end{{tabular}}

\smallskip
\noindent{{\footnotesize
$\star$ Recommended model.\enspace
Q90 target: $<{target_pct}$ relative error at 90th percentile.\enspace
KS: $p\geq 0.05$ = no overfitting detected.
}}

% ── 2. Accuracy Assessment ────────────────────────────────────────────────────
\section{{Accuracy Assessment (Q90 \& R\textsuperscript{{2}})}}
\label{{sec:accuracy}}

{acc_prose}

\vspace{{4pt}}
\renewcommand{{\arraystretch}}{{1.2}}
\begin{{tabular}}{{{col_spec}|}}
  \hline
  \textbf{{Output}} & {mdl_span} \\ \hline
  & {sub_hdr[3:]} \\ \hline
  {metrics_tex}
\end{{tabular}}

\smallskip
\noindent{{\footnotesize
  \colorbox{{r2green}}{{\ }} R\textsuperscript{{2}} $\geq 0.95$\enspace
  \colorbox{{r2amber}}{{\ }} R\textsuperscript{{2}} $\in[0.80, 0.95)$\enspace
  \colorbox{{failred}}{{\ }} R\textsuperscript{{2}} $< 0.80$ or Q90 fail\enspace
  Gap $=$ Q90 $-$ target (negative $\Rightarrow$ margin above target)
}}

% ── 3. Predicted vs True ──────────────────────────────────────────────────────
\section{{Predicted vs True (Scatter Plots, Test Set)}}
\label{{sec:scatter}}

Each scatter plot displays a random sample of up to 5\,000 test-set points.
The dashed diagonal represents perfect prediction ($\hat{{y}} = y$).
Systematic bias (points consistently above or below the diagonal) or
fan-shaped spread (heteroscedasticity) are indicators of model limitations.

{scatter_tex}

% ── 4. Overfitting Assessment ─────────────────────────────────────────────────
\section{{Overfitting Assessment (KS Test)}}
\label{{sec:ks}}

The Kolmogorov--Smirnov (KS) test compares the empirical distribution of
training residuals with that of test residuals.
$H_0$: both distributions are identical (no overfitting).
A p-value below 0.05 provides statistical evidence of overfitting.

\vspace{{4pt}}
{ks_prose}

\vspace{{4pt}}
\renewcommand{{\arraystretch}}{{1.2}}
\begin{{tabular}}{{{ks_col}}}
  \hline
  \textbf{{Output}} & {ks_hdrs} \\ \hline
  {ks_tex2}
\end{{tabular}}

\smallskip
\noindent{{\footnotesize
  \colorbox{{passgreen}}{{\ }} $p \geq 0.05$ --- residual distributions consistent\enspace
  \colorbox{{failred}}{{\ }} $p < 0.05$ --- overfitting detected
}}

% ── 5. Data Split Quality ─────────────────────────────────────────────────────
\section{{Data Split Quality (VTP Analysis)}}
\label{{sec:split}}

The Voronoi Tesselation Proximity (VTP) method evaluates whether the test
set is representative of, and well-covered by, the training distribution.
A low residual voxel proportion indicates that test points are not isolated
outside the training space.

\vspace{{4pt}}
{split_prose}

\vspace{{4pt}}
\renewcommand{{\arraystretch}}{{1.2}}
\begin{{tabular}}{{|l|r|c|}}
  \hline
  \textbf{{Metric}} & \textbf{{Value}} & \textbf{{Pass}} \\ \hline
  Residual voxel proportion   ($\leq 0.05$) & {rvp_val} & {rvp_icon} \\ \hline
  Valid test proportion                      & {vtp_val} & \\ \hline
  Phacking test proportion                   & {pht_val} & \\ \hline
  Isolated test proportion                   & {itp_val} & \\ \hline
  Chi\textsuperscript{{2}} p-value ($\geq 0.05$) & {chi_val} & {chi_icon} \\ \hline
\end{{tabular}}

% ── 6. Improvement Roadmap ────────────────────────────────────────────────────
\section{{Improvement Roadmap}}
\label{{sec:roadmap}}

The following actions are prioritised based on the failure analysis above.

\begin{{itemize}}[leftmargin=1.4em,itemsep=3pt,topsep=3pt]
  {detail_items}
\end{{itemize}}
"""


# ── Main builder ───────────────────────────────────────────────────────────────

def build_latex(d: dict, scatter_paths: dict) -> str:
    best = d['best_model']
    rows = _per_output(d, best)
    recs = _recommendations(d, best, rows, detailed=False)
    uc   = d['use_case']

    preamble = _PREAMBLE.replace('XUSECASEX', _esc(uc))

    return (
        preamble
        + r'\begin{document}' + '\n'
        + _part1(d, best, rows, recs, scatter_paths)
        + _separator(uc)
        + _part2(d, scatter_paths)
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
    scatter_paths = {}
    for m in d['models']:
        png = artifacts_dir / f'scatter_{m}.png'
        if png.exists():
            scatter_paths[m] = os.path.relpath(str(png), str(out_dir))

    tex = build_latex(d, scatter_paths)

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
        print('LaTeX source saved — compile manually with pdflatex or Overleaf.')
        sys.exit(1)


if __name__ == '__main__':
    main()
