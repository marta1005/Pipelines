"""
Generate a professional two-part PDF report from a Surrogate Factory metadata JSON.

Structure
---------
  Part 1 (≤ 2 pages): Executive Summary — winner model only, verdict,
                       pass/fail analysis with gap to target, recommendations.
  Separator page     : decorative divider.
  Part 2             : Technical Appendix — all models, full metrics,
                       scatter plots, KS test, split quality; with TOC hyperlinks.

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
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&',  r'\&'), ('%', r'\%'), ('$', r'\$'), ('#', r'\#'),
        ('_',  r'\_'), ('{', r'\{'), ('}', r'\}'),
        ('~',  r'\textasciitilde{}'), ('^', r'\textasciicircum{}'),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def _pct(v):
    return rf'{v * 100:.0f}\%' if v is not None else r'\textemdash'


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
    if passed:
        return rf'\cellcolor{{passgreen}}{val:.4f}'
    return rf'\cellcolor{{failred}}{val:.4f}'


def _gapcell(gap):
    if gap is None:
        return r'\textemdash'
    sign = '+' if gap > 0 else ''
    if gap > 0:
        return rf'\cellcolor{{failred}}{sign}{gap:.4f}'
    return rf'\cellcolor{{passgreen}}{sign}{gap:.4f}'


# ── Data extraction ────────────────────────────────────────────────────────────

def extract(meta_path: Path) -> dict:
    with open(meta_path) as f:
        root = json.load(f)['metadata']

    val  = root['Model_Validation']
    sel  = root['Model_Selection']
    trn  = root.get('Model_Training', {})
    part = root.get('Data_Partition', {}).get('data_split', {}).get('percentages', {})
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


# ── Analysis helpers ───────────────────────────────────────────────────────────

def _per_output(d, model):
    """Return list of dicts with per-output stats for `model`."""
    vres_map = {vr['output']: vr for vr in d['vres']}
    rows = []
    for o in d['outputs']:
        mres  = vres_map.get(o, {}).get('models', {}).get(model, {})
        r2    = d['scores'].get(model, {}).get(o, {}).get('R2')
        q90   = mres.get('score')
        ok    = mres.get('passed', False)
        gap   = round(q90 - d['q90_target'], 6) if q90 is not None else None
        rows.append(dict(output=o, r2=r2, q90=q90, passed=ok, gap=gap))
    return rows


def _recommendations(d, model, rows):
    """Generate dynamic recommendation bullets based on failure analysis."""
    target    = d['q90_target']
    failed    = [r for r in rows if not r['passed']]
    low_r2    = [r for r in rows if r['r2'] is not None and r['r2'] < 0.80]
    ks_fails  = [o for o in d['outputs']
                 if (d['dist'].get(model, {}).get(o) or 1.0) < 0.05]
    ks_all_ok = len(ks_fails) == 0
    best_q90_pass = d['q90_pass'].get(model, 0)
    n_out = len(d['outputs'])

    bullets = []

    if not failed:
        bullets.append(
            r'\textbf{Production-ready.} All outputs meet the Q90 accuracy requirement. '
            r'Deploy with confidence.'
        )
    else:
        slightly = [r for r in failed if r['gap'] is not None and r['gap'] < 0.05]
        severely = [r for r in failed if r['gap'] is not None and r['gap'] >= 0.05]

        if slightly:
            names = ', '.join(_esc(r['output']) for r in slightly)
            bullets.append(
                rf'\textbf{{Marginal failures}} (\textit{{{names}}}): '
                rf'Q90 is within {_pct(0.05)} of target. '
                r'Targeted data augmentation in under-represented flight conditions '
                r'is likely sufficient to close the gap.'
            )
        if severely:
            names = ', '.join(_esc(r['output']) for r in severely)
            bullets.append(
                rf'\textbf{{Significant failures}} (\textit{{{names}}}): '
                r'Error is substantially above target. '
                r'Investigate whether additional input features explain '
                r'the residual variance, or consider a more expressive model architecture.'
            )

    if low_r2:
        names = ', '.join(_esc(r['output']) for r in low_r2)
        bullets.append(
            rf'\textbf{{Low R\textsuperscript{{2}}}} (\textit{{{names}}}): '
            r'Coefficient of determination below 0.80 indicates high unexplained variance. '
            r'Review whether all physical drivers of these outputs are included as inputs.'
        )

    if not ks_all_ok:
        names = ', '.join(_esc(o) for o in ks_fails)
        bullets.append(
            rf'\textbf{{Overfitting detected}} (\textit{{{names}}}): '
            r'KS test rejects distributional equality between train and test residuals. '
            r'Apply stronger regularisation (dropout, weight decay) or reduce model complexity.'
        )
    else:
        bullets.append(
            r'\textbf{No overfitting detected.} KS test passes for all outputs: '
            r'residual distributions are consistent between train and test sets.'
        )

    return bullets


# ── LaTeX builders ─────────────────────────────────────────────────────────────

_PREAMBLE = r"""
\documentclass[9pt,a4paper]{article}
\usepackage[margin=1.8cm,top=2cm,bottom=2cm]{geometry}
\usepackage{booktabs,colortbl,xcolor,array,makecell,multicol}
\usepackage{amsmath,amssymb}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{fancyhdr,graphicx}
\usepackage[colorlinks=true,linkcolor=primary,urlcolor=primary,bookmarks=true]{hyperref}
\usepackage{parskip,enumitem}
\usepackage{titlesec}
\setlength{\parskip}{3pt}

\definecolor{passgreen}{RGB}{198,239,206}
\definecolor{failred}{RGB}{255,199,206}
\definecolor{warnamber}{RGB}{255,235,156}
\definecolor{r2green}{RGB}{198,239,206}
\definecolor{r2amber}{RGB}{255,235,156}
\definecolor{darkgreen}{RGB}{0,128,0}
\definecolor{primary}{RGB}{0,70,127}
\definecolor{sepgray}{RGB}{120,120,120}

\titleformat{\section}{\normalfont\normalsize\bfseries\color{primary}}{}{0em}{}[\vspace{-4pt}\rule{\linewidth}{0.4pt}]
\titlespacing{\section}{0pt}{8pt}{4pt}
% Lightweight heading safe inside minipage
\newcommand{\exhead}[1]{\medskip\noindent{\small\bfseries\color{primary}#1}\par\vspace{-5pt}\noindent\textcolor{primary}{\rule{\linewidth}{0.4pt}}\vspace{2pt}}

\pagestyle{fancy}\fancyhf{}
\lhead{\small\textcolor{primary}{\textbf{Surrogate Factory}}}
\chead{\small\textcolor{sepgray}{CONFIDENTIAL}}
\rhead{\small\textcolor{primary}{XUSECASEX}}
\lfoot{\small\textcolor{sepgray}{Surrogate Factory v2.2 --- Automated Validation Report}}
\rfoot{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}
"""


def _part1(d, best, rows, recommendations):
    """Executive Summary — max 2 pages."""
    uc          = _esc(d['use_case'])
    n_out       = len(d['outputs'])
    best_r2     = d['avg_r2'].get(best, 0)
    best_q90p   = d['q90_pass'].get(best, 0)
    best_ksp    = d['ks_pass'].get(best, 0)
    overall_ok  = best_q90p == n_out and best_ksp == n_out
    box_color   = 'passgreen' if overall_ok else 'warnamber'
    verdict     = 'ALL REQUIREMENTS MET' if overall_ok else 'REQUIREMENTS PARTIALLY MET'
    date_str    = datetime.today().strftime('%d %B %Y')
    target_pct  = _pct(d['q90_target'])
    inputs_str  = ', '.join(_esc(i) for i in d['inputs'])
    outputs_str = ', '.join(_esc(o) for o in d['outputs'])
    ptr         = _pct(d['pct_train'])
    pvl         = _pct(d['pct_val'])
    pts         = _pct(d['pct_test'])
    n_models    = len(d['models'])

    # ── Performance table (winner only) ───────────────────────────────────────
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
            + status + r' \\'
        )
    perf_table = '\n  '.join(perf_rows)

    # ── Recommendation bullets ─────────────────────────────────────────────────
    rec_items = '\n  '.join(r'\item ' + b for b in recommendations)

    return rf"""
% ════════════════════════════════════════════════════════════════
%  PART 1 — EXECUTIVE SUMMARY
% ════════════════════════════════════════════════════════════════
\thispagestyle{{fancy}}

% Title block
\begin{{center}}
  {{\large\bfseries\color{{primary}} Executive Summary --- Surrogate Model Validation}}\\[2pt]
  {{\normalsize\color{{sepgray}} Use case: \textbf{{{uc}}} \quad|\quad {date_str} \quad|\quad {n_models} model(s) evaluated}}
\end{{center}}
\vspace{{-4pt}}\noindent\rule{{\linewidth}}{{1pt}}\vspace{{2pt}}

% Verdict banner
\begin{{center}}
\colorbox{{{box_color}}}{{\parbox{{0.55\linewidth}}{{\centering\small
  \textbf{{{verdict}}}\\[3pt]
  Recommended model: \textbf{{{_esc(best)}}}\enspace|\enspace
  Avg R\textsuperscript{{2}}: \textbf{{{best_r2:.4f}}}\enspace|\enspace
  Q90: \textbf{{{best_q90p}/{n_out}}}\enspace|\enspace
  KS: \textbf{{{best_ksp}/{n_out}}}}}}}
\end{{center}}
\vspace{{4pt}}

% ── Overview + Legend side by side ────────────────────────────────────────────
\begin{{minipage}}[t]{{0.52\linewidth}}
\exhead{{Project Overview}}
\renewcommand{{\arraystretch}}{{1.1}}
\begin{{tabular}}{{@{{}}lp{{0.58\linewidth}}}}
  \textbf{{Use case}}    & {uc} \\
  \textbf{{Inputs}}      & {inputs_str} \\
  \textbf{{Outputs}}     & {n_out}: {outputs_str} \\
  \textbf{{Split}}       & Train {ptr} / Val {pvl} / Test {pts} \\
  \textbf{{Requirement}} & Q90 $<$ {target_pct} per output \\
\end{{tabular}}
\end{{minipage}}
\hfill
\begin{{minipage}}[t]{{0.44\linewidth}}
\exhead{{Reading the Table Below}}
\renewcommand{{\arraystretch}}{{1.1}}
\begin{{tabular}}{{@{{}}ll}}
  R\textsuperscript{{2}} & Goodness of fit (higher = better) \\
  Q90                    & 90th pct.\ relative error \\
  Gap                    & Q90 $-$ target ($<0$ = margin) \\
  \cellcolor{{passgreen}}PASS & Q90 $<$ {target_pct} \\
  \cellcolor{{failred}}FAIL   & Q90 $\geq$ {target_pct} \\
  \cellcolor{{r2green}}       & R\textsuperscript{{2}} $\geq 0.95$ \\
  \cellcolor{{r2amber}}       & $0.80 \leq$ R\textsuperscript{{2}} $< 0.95$ \\
  \cellcolor{{failred}}       & R\textsuperscript{{2}} $< 0.80$ \\
\end{{tabular}}
\end{{minipage}}

\vspace{{4pt}}

% ── Performance table ─────────────────────────────────────────────────────────
\section{{Model Performance --- \textbf{{{_esc(best)}}} (Test Set)}}

\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabular}}{{lrrrrrc}}
  \toprule
  \textbf{{Output}} &
  \textbf{{R\textsuperscript{{2}}}} &
  \textbf{{Q90}} &
  \textbf{{Target}} &
  \textbf{{Gap}} &
  \textbf{{Status}} \\
  \midrule
  {perf_table}
  \bottomrule
\end{{tabular}}

\vspace{{4pt}}

% ── Analysis ─────────────────────────────────────────────────────────────────
\section{{Analysis \& Recommendations}}

\begin{{itemize}}[leftmargin=1.2em,itemsep=2pt,topsep=2pt]
  {rec_items}
\end{{itemize}}

\vspace{{4pt}}\noindent\textcolor{{sepgray}}{{\small
  \textit{{Full technical details --- scatter plots, KS overfitting test, split quality ---
  in the \hyperref[sec:appendix]{{Technical Appendix}} (page \pageref{{sec:appendix}}).}}
}}
"""


def _separator(uc):
    return rf"""
% ════════════════════════════════════════════════════════════════
%  SEPARATOR
% ════════════════════════════════════════════════════════════════
\clearpage
\thispagestyle{{empty}}
\vspace*{{\fill}}
\begin{{center}}
  \textcolor{{sepgray}}{{\rule{{0.3\linewidth}}{{0.6pt}}}}\\[12pt]
  {{\Large\bfseries\color{{primary}} Technical Appendix}}\\[6pt]
  {{\normalsize\color{{sepgray}} {_esc(uc)} --- Full Validation Report}}\\[12pt]
  \textcolor{{sepgray}}{{\rule{{0.3\linewidth}}{{0.6pt}}}}
\end{{center}}
\vspace*{{\fill}}
\clearpage
"""


def _part2(d, scatter_paths):
    """Technical Appendix — all models, full metrics, plots, KS, split."""
    models     = d['models']
    outputs    = d['outputs']
    scores     = d['scores']
    dist       = d['dist']
    vres       = d['vres']
    split      = d['split']
    n_out      = len(outputs)
    n_mdl      = len(models)
    target_pct = _pct(d['q90_target'])

    # ── All-models comparison table ───────────────────────────────────────────
    comp_rows = []
    for m in models:
        q90p = d['q90_pass'].get(m, 0)
        ksp  = d['ks_pass'].get(m, 0)
        avg  = d['avg_r2'].get(m, 0)
        best = m == d['best_model']
        b    = r'\bfseries' if best else ''
        star = r' $\star$' if best else ''
        comp_rows.append(
            rf'  {{{b} {_esc(m)}{star}}} & {{{b} {avg:.4f}}} & {{{b} {q90p}/{n_out}}} & {{{b} {ksp}/{n_out}}} \\'
        )
    comp_tex = '\n'.join(comp_rows)

    # ── Full metrics table (all models × all outputs) ─────────────────────────
    vres_map   = {vr['output']: vr for vr in vres}
    col_spec   = 'l' + 'rrr' * n_mdl
    mdl_span   = ' & '.join(
        rf'\multicolumn{{3}}{{c}}{{\textbf{{{_esc(m)}}}}}'
        for m in models
    )
    sub_hdr    = (r' & R\textsuperscript{2} & Q90 & Gap') * n_mdl
    metric_rows = []
    for o in outputs:
        cells = [_esc(o)]
        for m in models:
            r2    = scores.get(m, {}).get(o, {}).get('R2')
            mres  = vres_map.get(o, {}).get('models', {}).get(m, {})
            q90   = mres.get('score')
            ok    = mres.get('passed', False)
            gap   = round(q90 - d['q90_target'], 6) if q90 is not None else None
            cells += [_r2cell(r2), _q90cell(q90, ok), _gapcell(gap)]
        metric_rows.append(' & '.join(cells) + r' \\')
    metrics_tex = '\n  '.join(metric_rows)

    # ── KS test table ─────────────────────────────────────────────────────────
    ks_col  = 'l' + 'r' * n_mdl
    ks_hdrs = ' & '.join(r'\textbf{' + _esc(m) + '}' for m in models)
    ks_rows = []
    for o in outputs:
        cells = [_esc(o)]
        for m in models:
            pval = dist.get(m, {}).get(o)
            if pval is None:
                cells.append(r'\textemdash')
            else:
                cells.append(
                    rf'\cellcolor{{passgreen}}{pval:.3f}' if pval >= 0.05
                    else rf'\cellcolor{{failred}}{pval:.3f}'
                )
        ks_rows.append(' & '.join(cells) + r' \\')
    ks_tex = '\n  '.join(ks_rows)

    # ── Split quality ──────────────────────────────────────────────────────────
    def _sv(key, fmt='.3f'):
        v = split.get(key)
        return f'{v:{fmt}}' if v is not None else r'\textemdash'

    def _sicon(key, op, thresh):
        v = split.get(key)
        if v is None:
            return r'\textemdash'
        ok = (v <= thresh) if op == '<=' else (v >= thresh)
        return r'\cellcolor{passgreen}\checkmark' if ok else r'\cellcolor{failred}\times'

    rvp_val  = _sv('residual_voxel_proportion')
    rvp_icon = _sicon('residual_voxel_proportion', '<=', 0.05)
    vtp_val  = _sv('valid_test_proportion')
    pht_val  = _sv('phacking_test_proportion')
    itp_val  = _sv('isolated_test_proportion')
    chi_val  = _sv('chi_squared_pvalue', '.4f')
    chi_icon = _sicon('chi_squared_pvalue', '>=', 0.05)

    # ── Scatter plots ──────────────────────────────────────────────────────────
    scatter_tex = ''
    for m in models:
        path = scatter_paths.get(m, '')
        if path:
            scatter_tex += (
                rf'\subsection*{{{_esc(m)}}}' + '\n'
                r'\begin{center}' + '\n'
                r'  \includegraphics[width=0.78\textwidth]{' + path + r'}' + '\n'
                r'\end{center}' + '\n\n'
            )

    return rf"""
% ════════════════════════════════════════════════════════════════
%  PART 2 — TECHNICAL APPENDIX
% ════════════════════════════════════════════════════════════════
\label{{sec:appendix}}
\phantomsection
\addcontentsline{{toc}}{{section}}{{Technical Appendix}}

\begin{{center}}
  {{\normalsize\bfseries\color{{primary}} Technical Appendix --- Contents}}
\end{{center}}
\tableofcontents
\vspace{{8pt}}

% ── 1. Model Comparison ───────────────────────────────────────────────────────
\section{{Model Comparison (Test Set)}}
\label{{sec:comparison}}

\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabular}}{{lrrr}}
  \toprule
  \textbf{{Model}} & \textbf{{Avg R\textsuperscript{{2}}}} &
  \textbf{{Q90 pass ({n_out} outputs)}} & \textbf{{KS pass ({n_out} outputs)}} \\
  \midrule
{comp_tex}
  \bottomrule
\end{{tabular}}

\smallskip
\noindent\textit{{$\star$ = recommended model.\enspace
Q90 target: $<{target_pct}$ relative error.\enspace
KS: $p\geq 0.05$ = no overfitting.}}

% ── 2. Full Metrics per Output ────────────────────────────────────────────────
\section{{Full Metrics per Output}}
\label{{sec:metrics}}

\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabular}}{{{col_spec}}}
  \toprule
  \textbf{{Output}} & {mdl_span} \\
  & {sub_hdr[3:]} \\
  \midrule
  {metrics_tex}
  \bottomrule
\end{{tabular}}

\smallskip
\noindent
\colorbox{{passgreen}}{{pass}}\enspace Q90 $<{target_pct}$\quad
\colorbox{{failred}}{{fail / low R\textsuperscript{{2}}}}\quad
\colorbox{{r2amber}}{{R\textsuperscript{{2}} $\in [0.80,\,0.95)$}}\quad
Gap $= $ Q90 $-$ target (negative $\Rightarrow$ margin).

% ── 3. Predicted vs True ─────────────────────────────────────────────────────
\section{{Predicted vs True (Scatter Plots, Test Set)}}
\label{{sec:scatter}}

\noindent Each plot shows a sample of up to 5\,000 test points.
The dashed line represents perfect prediction ($\hat{{y}}=y$).

{scatter_tex}

% ── 4. Overfitting Check — KS Test ───────────────────────────────────────────
\section{{Overfitting Check --- KS Test p-values}}
\label{{sec:ks}}

\noindent$H_0$: residuals on train and test sets follow the same distribution.
\textbf{{Rejection ($p<0.05$) indicates overfitting.}}

\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabular}}{{{ks_col}}}
  \toprule
  \textbf{{Output}} & {ks_hdrs} \\
  \midrule
  {ks_tex}
  \bottomrule
\end{{tabular}}

% ── 5. Train/Test Split Quality ───────────────────────────────────────────────
\section{{Train / Test Split Quality}}
\label{{sec:split}}

\renewcommand{{\arraystretch}}{{1.15}}
\begin{{tabular}}{{llc}}
  \toprule
  \textbf{{Metric}} & \textbf{{Value}} & \textbf{{Pass}} \\
  \midrule
  Residual voxel proportion  & {rvp_val} & {rvp_icon} \\
  Valid test proportion      & {vtp_val} & \\
  Phacking test proportion   & {pht_val} & \\
  Isolated test proportion   & {itp_val} & \\
  Chi\textsuperscript{{2}} p-value & {chi_val} & {chi_icon} \\
  \bottomrule
\end{{tabular}}

\smallskip
\noindent\textit{{Residual voxel proportion $\leq 0.05$ ensures test points cover
the training space. Chi\textsuperscript{{2}} $p \geq 0.05$ confirms the split is
statistically unbiased.}}
"""


# ── Main builder ───────────────────────────────────────────────────────────────

def build_latex(d: dict, scatter_paths: dict) -> str:
    best  = d['best_model']
    rows  = _per_output(d, best)
    recs  = _recommendations(d, best, rows)
    uc    = d['use_case']

    preamble = _PREAMBLE.replace('XUSECASEX', _esc(uc))
    body = (
        r'\begin{document}' + '\n'
        + _part1(d, best, rows, recs)
        + _separator(uc)
        + _part2(d, scatter_paths)
        + '\n' + r'\end{document}'
    )
    return preamble + body


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate a professional LaTeX/PDF report from SF metadata JSON.')
    parser.add_argument('metadata', help='Path to metadata_*.json')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: same folder as metadata)')
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    out_dir   = Path(args.output).resolve() if args.output else meta_path.parent.resolve()
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
        print('tectonic STDERR:\n', result.stderr[-2000:])
        print('LaTeX source saved — compile manually with pdflatex or Overleaf.')
        sys.exit(1)


if __name__ == '__main__':
    main()
