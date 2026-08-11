"""
Generate professional multi-part PDF + HTML validation report.

Structure
---------
  Part 1  Executive Summary  (≤2 pages, winner model only)
  Part 2  Technical Review   (all models, TOC, scatter/ratio, KS, split, roadmap)
  ─── separator ─────────────────────────────────────────────────────────────────
  Part 3  Deep Analysis — winner model only, mirrors validation_template.ipynb
             3.1 Data Overview          inputs + outputs stats, histograms, CDFs
             3.2 Train-Test Split       double histograms + KS/AD table
             3.3 Error Quantification   residue + abs-error histograms + CDFs
             3.4 P(E|X)                 scatter + violin per output vs inputs
             3.5 P(E|Y)                 scatter + violin per output vs true output
             3.6 Uncertainty            sigma coverage table
          HTML export alongside PDF.
"""
import argparse, base64, json, os, subprocess, sys
from datetime import datetime
from pathlib import Path


# ── LaTeX helpers ──────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    for old, new in [
        ('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
        ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
        ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}'),
    ]:
        s = s.replace(old, new)
    return s

def _pct(v, d=0):
    return rf'{v*100:.{d}f}\%' if v is not None else r'\textemdash'

def _r2cell(r2):
    if r2 is None: return r'\textemdash'
    if r2 >= 0.95: return rf'\cellcolor{{r2green}}{r2:.4f}'
    if r2 >= 0.80: return rf'\cellcolor{{r2amber}}{r2:.4f}'
    return rf'\cellcolor{{failred}}{r2:.4f}'

def _q90cell(val, passed):
    if val is None: return r'\textemdash'
    return rf'\cellcolor{{{"passgreen" if passed else "failred"}}}{val:.4f}'

def _gapcell(gap):
    if gap is None: return r'\textemdash'
    s = '+' if gap > 0 else ''
    return rf'\cellcolor{{{"failred" if gap>0 else "passgreen"}}}{s}{gap:.4f}'

def _kscell(pval):
    if pval is None: return r'\textemdash'
    return rf'\cellcolor{{{"passgreen" if pval>=0.05 else "failred"}}}{pval:.3f}'

_LEGEND = (
    r'{\footnotesize'
    r' \colorbox{r2green}{\ } R\textsuperscript{2}$\!\geq\!0.95$\enspace'
    r' \colorbox{r2amber}{\ } R\textsuperscript{2}$\!\in[0.80,0.95)$\enspace'
    r' \colorbox{failred}{\ } R\textsuperscript{2}$\!<\!0.80$ or Q90 fail\enspace'
    r' \colorbox{passgreen}{\ } pass\enspace'
    r' Gap$=$Q90$-$target ($<\!0\Rightarrow$ margin)}'
)
_KS_LEGEND = (
    r'{\footnotesize'
    r' \colorbox{passgreen}{\ } $p\geq 0.05$ --- no overfitting\enspace'
    r' \colorbox{failred}{\ } $p<0.05$ --- overfitting detected}'
)

def _tbl_with_legend(tbl_tex, legend_tex):
    """Keep table+legend on the same page; legend always below the table."""
    return (r'\begin{minipage}{\linewidth}' + '\n'
            + tbl_tex + '\n'
            + r'\par\vspace{4pt}' + '\n'
            + legend_tex + '\n'
            + r'\end{minipage}')


# ── Data extraction ────────────────────────────────────────────────────────────

def extract(meta_path: Path) -> dict:
    with open(meta_path) as f:
        root = json.load(f)['metadata']
    if 'Model_Validation' not in root:
        raise SystemExit(
            f"ERROR: '{meta_path.name}' contains no Model_Validation section, so there\n"
            f"       are no results to report on.\n\n"
            f"       This happens when SF_9 did not run to completion — the validation\n"
            f"       stages write their results into the workflow metadata, but they are\n"
            f"       only persisted by the final `workflow.save_metadata()` cell.\n\n"
            f"       Fix: run SF_9_Model_Validation.ipynb through to the Save cell, then\n"
            f"       re-run this report.\n\n"
            f"       Sections currently present: {', '.join(sorted(root)) or '(none)'}"
        )
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
    acc     = req.get('accuracy', [])
    q90_target = acc[0]['value'] if acc else 0.10
    avg_r2 = {}
    for m in models:
        r2s = [scores.get(m, {}).get(o, {}).get('R2') for o in outputs]
        r2s = [v for v in r2s if v is not None]
        avg_r2[m] = sum(r2s) / len(r2s) if r2s else 0
    best = max(avg_r2, key=avg_r2.get) if avg_r2 else (models[0] if models else 'N/A')
    q90_pass = {m: 0 for m in models}
    for vr in vres:
        for m, res in vr.get('models', {}).items():
            if res.get('passed'):
                q90_pass[m] = q90_pass.get(m, 0) + 1
    ks_pass = {m: sum(1 for v in dist.get(m, {}).values() if v >= 0.05) for m in models}
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

def _recs(d, model, rows, detailed=False):
    failed  = [r for r in rows if not r['passed']]
    low_r2  = [r for r in rows if r['r2'] is not None and r['r2'] < 0.80]
    ks_fail = [o for o in d['outputs'] if (d['dist'].get(model, {}).get(o) or 1.0) < 0.05]
    bullets = []
    if not failed:
        bullets.append(r'\textbf{Production-ready.} All outputs satisfy Q90. Deploy with confidence.')
    else:
        slight = [r for r in failed if r['gap'] is not None and r['gap'] < 0.05]
        severe = [r for r in failed if r['gap'] is not None and r['gap'] >= 0.05]
        if slight:
            names = ', '.join(_esc(r['output']) for r in slight)
            ex = r' Collect data near under-represented flight conditions.' if detailed else ''
            bullets.append(rf'\textbf{{Marginal failures}} (\textit{{{names}}}): Q90 within 5\% of target. Targeted data augmentation likely sufficient.{ex}')
        if severe:
            names = ', '.join(_esc(r['output']) for r in severe)
            ex = r' Review input features and consider a more expressive architecture.' if detailed else ''
            bullets.append(rf'\textbf{{Significant failures}} (\textit{{{names}}}): Q90 exceeds target by $\geq$5\%. Investigate inputs and model capacity.{ex}')
    if low_r2:
        names = ', '.join(_esc(r['output']) for r in low_r2)
        bullets.append(rf'\textbf{{Low R\textsuperscript{{2}}}} (\textit{{{names}}}): Below 0.80 --- high unexplained variance.')
    if ks_fail:
        names = ', '.join(_esc(o) for o in ks_fail)
        ex = r' Increase regularisation or reduce model capacity.' if detailed else ''
        bullets.append(rf'\textbf{{Overfitting}} (\textit{{{names}}}): KS test rejects equal residual distributions.{ex}')
    else:
        bullets.append(r'\textbf{No overfitting.} KS confirms consistent residuals on train and test.')
    return bullets


# ── Figure sizing ──────────────────────────────────────────────────────────────
# width alone is not enough: a tall, narrow figure (more inputs than outputs)
# scaled to the full text width grows far past the bottom of the page. Capping
# the height as well, with keepaspectratio, makes the image fit inside whichever
# dimension binds first — so nothing can ever run off the sheet.
_GFX_MAXH = r'height=0.80\textheight'
_GFX_FIT  = r'width=\linewidth,' + _GFX_MAXH + r',keepaspectratio'


# ── Preamble ───────────────────────────────────────────────────────────────────

_PREAMBLE = r"""\documentclass[9pt,a4paper]{article}
\usepackage[margin=1.8cm,top=2.0cm,bottom=2.0cm]{geometry}
\usepackage{booktabs,colortbl,xcolor,array,makecell,hhline}
\usepackage{amsmath,amssymb}
\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}
\usepackage{fancyhdr,graphicx}
\usepackage[colorlinks=true,linkcolor=primary,urlcolor=primary,
            bookmarks=true,pdfborder={0 0 0}]{hyperref}
\usepackage{parskip,enumitem,titlesec,needspace}
\setlength{\parskip}{2pt}
\definecolor{passgreen}{RGB}{198,239,206}
\definecolor{failred}{RGB}{255,199,206}
\definecolor{warnamber}{RGB}{255,235,156}
\definecolor{r2green}{RGB}{198,239,206}
\definecolor{r2amber}{RGB}{255,235,156}
\definecolor{primary}{RGB}{0,70,127}
\definecolor{sepgray}{RGB}{130,130,130}
\titleformat{\section}{\normalsize\bfseries\color{primary}}{}{0em}{}
            [\vspace{-4pt}\textcolor{primary}{\rule{\linewidth}{0.5pt}}]
\titlespacing{\section}{0pt}{16pt}{6pt}
\newcommand{\exhead}[1]{%
  \medskip\noindent{\small\bfseries\color{primary}#1}%
  \par\vspace{-5pt}\noindent\textcolor{primary}{\rule{\linewidth}{0.4pt}}\vspace{2pt}}
\pagestyle{fancy}\fancyhf{}
\lhead{\small\textcolor{primary}{\textbf{Surrogate Factory}}}
\chead{\small\textcolor{sepgray}{CONFIDENTIAL}}
\rhead{\small\textcolor{primary}{XUSECASEX}}
\lfoot{\small\textcolor{sepgray}{Surrogate Factory v2.2 --- Automated Validation Report}}
\rfoot{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}\renewcommand{\footrulewidth}{0.4pt}
"""


# ── Banner (Part 1) ────────────────────────────────────────────────────────────

def _banner(d, best, rows):
    n_out    = len(d['outputs'])
    best_r2  = d['avg_r2'].get(best, 0)
    q90p     = d['q90_pass'].get(best, 0)
    ksp      = d['ks_pass'].get(best, 0)
    ok       = (q90p == n_out) and (ksp == n_out)
    box_c    = 'passgreen' if ok else 'warnamber'
    verdict  = 'ALL REQUIREMENTS MET' if ok else 'REQUIREMENTS PARTIALLY MET'
    r2c      = 'r2green' if best_r2 >= 0.95 else ('r2amber' if best_r2 >= 0.80 else 'failred')
    r2l      = 'Excellent' if best_r2 >= 0.95 else ('Acceptable' if best_r2 >= 0.80 else 'Poor')
    q90_p    = [_esc(r['output']) for r in rows if r['passed']]
    q90_f    = [_esc(r['output']) for r in rows if not r['passed']]
    q90c     = 'passgreen' if q90p == n_out else ('warnamber' if q90p > 0 else 'failred')
    q90l     = 'All outputs pass' if q90p == n_out else ('Partially met' if q90p > 0 else 'No output passes')
    ks_p     = [_esc(o) for o in d['outputs'] if (d['dist'].get(best, {}).get(o) or 1.0) >= 0.05]
    ks_f     = [_esc(o) for o in d['outputs'] if (d['dist'].get(best, {}).get(o) or 1.0) < 0.05]
    ksc      = 'passgreen' if ksp == n_out else ('warnamber' if ksp > n_out // 2 else 'failred')
    ksl      = 'No overfitting' if ksp == n_out else ('Minor overfitting' if ksp > n_out // 2 else 'Overfitting present')

    def _row(color, title, label, pass_names, fail_names):
        lines = [rf'{{\small\bfseries {title}}} \enspace---\enspace {{\small {label}}}']
        if pass_names:
            lines.append(rf'{{\footnotesize Pass: {", ".join(pass_names)}}}')
        if fail_names:
            lines.append(rf'{{\footnotesize\color{{red}} Fail: {", ".join(fail_names)}}}')
        body = r'\\[1pt]'.join(lines)
        return (rf'\colorbox{{{color}}}{{\begin{{minipage}}{{0.97\linewidth}}'
                rf'\vspace{{2pt}}\centering {body}\vspace{{2pt}}\end{{minipage}}}}')

    best_esc = _esc(best)
    r2_str   = f'{best_r2:.4f}'
    return (
        r'\begin{center}' + '\n'
        rf'\colorbox{{{box_c}}}{{\begin{{minipage}}{{0.66\linewidth}}' + '\n'
        r'  \vspace{5pt}\centering' + '\n'
        rf'  {{\normalsize\bfseries {verdict}}}\\[4pt]' + '\n'
        rf'  {{\small\bfseries Recommended model:}} {{\small {best_esc}}}\\[5pt]' + '\n'
        + _row(r2c, rf'Avg R\textsuperscript{{2}}: {r2_str}', r2l, [], []) + r'\\[2pt]' + '\n'
        + _row(q90c, 'Q90 Accuracy', q90l, q90_p, q90_f) + '\n'
        r'  \vspace{5pt}' + '\n'
        r'\end{minipage}}' + '\n'
        r'\end{center}'
    )


# ── Executive Summary ──────────────────────────────────────────────────────────

def _part1(d, best, rows, scatter_paths, paths, out_dir):
    uc          = _esc(d['use_case'])
    n_out       = len(d['outputs'])
    n_models    = len(d['models'])
    date_str    = datetime.today().strftime('%d %B %Y')
    inputs_str  = ', '.join(_esc(i) for i in d['inputs'])
    outputs_str = ', '.join(_esc(o) for o in d['outputs'])
    target_pct  = _pct(d['q90_target'])
    ptr, pvl, pts = _pct(d.get('pct_train')), _pct(d.get('pct_val')), _pct(d.get('pct_test'))
    best_esc    = _esc(best)
    models      = d['models']
    outputs     = d['outputs']
    scores      = d['scores']
    dist        = d['dist']
    vres        = d['vres']
    split       = d['split']
    vres_map    = {vr['output']: vr for vr in vres}
    _star       = {m: (r'\ $\star$' if m == best else '') for m in models}

    def _fig_es(name):
        p = paths.get(name, '')
        if not p: return ''
        rp = os.path.relpath(p, str(out_dir))
        return (r'\begin{center}\includegraphics[' + _GFX_FIT + r']{'
                + rp + r'}\end{center}' + '\n')

    def _fig_es_multi(name):
        """Emit a figure plus any `<name>_2`, `_3`, ... continuation chunks."""
        out, i = _fig_es(name), 2
        while f'{name}_{i}' in paths:
            out += _fig_es(f'{name}_{i}')
            i += 1
        return out

    # ── Winner scatter ─────────────────────────────────────────────────────────
    winner_scatter_tex = ''
    p = scatter_paths.get(best, '')
    if p:
        winner_scatter_tex = (
            r'\begin{center}\includegraphics[' + _GFX_FIT + r']{'
            + p + r'}\end{center}' + '\n'
        )

    # ── Model comparison table (both models) ──────────────────────────────────
    comp_rows = []
    for m in models:
        b = r'\bfseries' if m == best else ''
        comp_rows.append(
            rf'  {{{b} {_esc(m)}{_star[m]}}} & {{{b} {d["avg_r2"].get(m,0):.4f}}}'
            rf' & {{{b} {d["q90_pass"].get(m,0)}/{n_out}}}'
            rf' & {{{b} {d["ks_pass"].get(m,0)}/{n_out}}} \\ \hline'
        )
    comp_tex = '\n'.join(comp_rows)
    comp_tbl = (
        r'\renewcommand{\arraystretch}{1.2}' + '\n'
        r'\begin{tabular}{|l|r|r|r|}' + '\n'
        r'  \hline' + '\n'
        rf'  \textbf{{Model}} & \textbf{{Avg R\textsuperscript{{2}}}} & \textbf{{Q90 pass ({n_out})}} & \textbf{{KS pass ({n_out})}} \\ \hline' + '\n'
        + comp_tex + '\n'
        r'\end{tabular}'
    )
    comp_legend = (
        r'{\footnotesize $\star$ Recommended model.\enspace '
        rf'Q90 target: $<{target_pct}$.\enspace KS: $p\geq 0.05$ = no overfitting.}}'
    )

    # ── Per-output metrics table (all models) ──────────────────────────────────
    col_spec  = 'l' + '|rrr' * len(models)
    mdl_span  = ' & '.join(
        rf'\multicolumn{{3}}{{c|}}{{\textbf{{{_esc(m)}}}{_star[m]}}}' for m in models)
    sub_hdr   = (r' & R\textsuperscript{2} & Q90 & Gap') * len(models)
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
    metrics_tbl = (
        r'\renewcommand{\arraystretch}{1.2}' + '\n'
        rf'\begin{{tabular}}{{{col_spec}|}}' + '\n'
        r'  \hline' + '\n'
        rf'  \textbf{{Output}} & {mdl_span} \\ \hline' + '\n'
        rf'  & {sub_hdr[3:]} \\ \hline' + '\n'
        f'  {metrics_tex}\n'
        r'\end{tabular}'
    )

    # ── KS table (winner model) ────────────────────────────────────────────────
    ks_cells = []
    for o in outputs:
        pval = dist.get(best, {}).get(o)
        ks_cells.append(_esc(o) + ' & ' + _kscell(pval) + r' \\ \hline')
    ks_tex = '\n  '.join(ks_cells)
    ks_tbl = (
        r'\renewcommand{\arraystretch}{1.15}' + '\n'
        r'\begin{tabular}{|l|r|}' + '\n'
        r'  \hline\textbf{Output} & \textbf{KS p-value} \\ \hline' + '\n'
        f'  {ks_tex}\n'
        r'\end{tabular}'
    )

    # ── Data Split Quality table ───────────────────────────────────────────────
    def _sv(key, fmt='.3f'):
        v = split.get(key)
        return f'{v:{fmt}}' if v is not None else r'\textemdash'

    def _si(key, op, th):
        v = split.get(key)
        if v is None: return r'\textemdash'
        ok = (v <= th) if op == '<=' else (v >= th)
        return r'\cellcolor{passgreen}$\checkmark$' if ok else r'\cellcolor{failred}$\times$'

    split_tbl = (
        r'\begin{minipage}{\linewidth}' + '\n'
        r'\renewcommand{\arraystretch}{1.2}' + '\n'
        r'\begin{tabular}{|l|r|c|}' + '\n'
        r'  \hline\textbf{Metric} & \textbf{Value} & \textbf{Pass} \\ \hline' + '\n'
        rf'  Residual voxel proportion ($\leq\!0.05$) & {_sv("residual_voxel_proportion")} & {_si("residual_voxel_proportion","<=",0.05)} \\ \hline' + '\n'
        rf'  Valid test proportion & {_sv("valid_test_proportion")} & \\ \hline' + '\n'
        rf'  Phacking test proportion & {_sv("phacking_test_proportion")} & \\ \hline' + '\n'
        rf'  Isolated test proportion & {_sv("isolated_test_proportion")} & \\ \hline' + '\n'
        rf'  Chi\textsuperscript{{2}} p-value ($\geq\!0.05$) & {_sv("chi_squared_pvalue",".4f")} & {_si("chi_squared_pvalue",">=",0.05)} \\ \hline' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{minipage}'
    )

    return (
        '% ==== EXECUTIVE SUMMARY ====\n'
        r'\thispagestyle{fancy}' + '\n\n'
        r'\begin{center}' + '\n'
        rf'  {{\large\bfseries\color{{primary}} Executive Summary --- Surrogate Model Validation Report}}\\[2pt]' + '\n'
        rf'  {{\small\color{{sepgray}} Use case: \textbf{{{uc}}} \quad|\quad {date_str} \quad|\quad {n_models} model(s)}}' + '\n'
        r'\end{center}' + '\n'
        r'\vspace{-4pt}\noindent\rule{\linewidth}{1pt}\vspace{3pt}' + '\n\n'

        # 1. Requirements
        + _banner(d, best, rows) + '\n\n'

        # 2. Project Overview
        r'\section{Project Overview}' + '\n'
        r'\renewcommand{\arraystretch}{1.2}' + '\n'
        r'\begin{tabular}{|l|p{0.60\linewidth}|}' + '\n'
        r'  \hline' + '\n'
        rf'  \textbf{{Use case}}           & {uc} \\ \hline' + '\n'
        rf'  \textbf{{Inputs}}             & {inputs_str} \\ \hline' + '\n'
        rf'  \textbf{{Outputs}}            & {n_out}: {outputs_str} \\ \hline' + '\n'
        rf'  \textbf{{Train / Val / Test}} & {ptr} / {pvl} / {pts} \\ \hline' + '\n'
        rf'  \textbf{{Accuracy target}}    & Q90 $<$ {target_pct} relative error per output \\ \hline' + '\n'
        r'\end{tabular}' + '\n\n'

        # 3. Variable correlation scatter
        r'\section{Variable Correlation --- Input vs Output}' + '\n'
        + _fig_es_multi('data_scatter_vars')

        # 4. Model Selection
        + r'\section{Model Selection}' + '\n'
        + _tbl_with_legend(comp_tbl, comp_legend) + '\n\n'

        # 5. Accuracy Assessment
        r'\section{Accuracy Assessment (Q90 \& R\textsuperscript{2})}' + '\n'
        + _tbl_with_legend(metrics_tbl, _LEGEND) + '\n\n'

        # 6. Predicted vs True — winner model
        rf'\section{{Predicted vs True --- {best_esc}}}' + '\n'
        + winner_scatter_tex + '\n\n'

        # 7. Data Split Quality
        r'\section{Data Split Quality}' + '\n'
        + split_tbl + '\n\n'

        # 8. Overfitting Check
        rf'\section{{Overfitting Check (KS Test) --- {best_esc}}}' + '\n'
        + _tbl_with_legend(ks_tbl, _KS_LEGEND) + '\n'
    )


# ── Separators ────────────────────────────────────────────────────────────────

def _separator_inner(uc_esc, title, subtitle):
    """Full-page separator (already esc'd strings)."""
    return (
        r'\clearpage' + '\n'
        r'\thispagestyle{empty}' + '\n'
        r'\vspace*{\fill}' + '\n'
        r'\begin{center}' + '\n'
        r'  \textcolor{sepgray}{\rule{0.40\linewidth}{1pt}}\\[20pt]' + '\n'
        rf'  {{\Huge\bfseries\color{{primary}} {title}}}\\[10pt]' + '\n'
        rf'  {{\large\color{{sepgray}} {uc_esc}}}\\[6pt]' + '\n'
        rf'  {{\normalsize\color{{sepgray}} {subtitle}}}\\[20pt]' + '\n'
        r'  \textcolor{sepgray}{\rule{0.40\linewidth}{1pt}}' + '\n'
        r'\end{center}' + '\n'
        r'\vspace*{\fill}' + '\n'
        r'\clearpage' + '\n'
    )

def _separator(uc):
    return _separator_inner(_esc(uc), 'Deep Analysis', 'Winner Model --- Full Validation')


# ── Analysis plots ─────────────────────────────────────────────────────────────

def _sturges(n):
    import math
    return max(5, int(math.ceil(math.log2(n) + 1)))

def _axes_flat(axes, nrows, ncols):
    """Always return a flat list of axes, regardless of subplot shape."""
    import numpy as np
    arr = np.array(axes)
    return arr.flatten().tolist()

def _violin_grid(fig, axes_flat, n_out, data_list, outputs, title, ylabel):
    """Fill a grid of violin plots; data_list[i] is list of arrays for output i."""
    for i, o in enumerate(outputs):
        ax = axes_flat[i]
        groups = [g for g in data_list[i] if len(g) >= 3]
        if groups:
            parts = ax.violinplot(groups, positions=range(len(groups)), showmedians=True)
            for pc in parts.get('bodies', []):
                pc.set_alpha(0.6)
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_title(o, fontsize=8)
        ax.set_ylabel(ylabel if i % max(1, len(outputs) // 4 + 1) == 0 else '', fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(n_out, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(title, fontsize=10)


def _analysis_plots(best, csv_dir, plots_dir, q90_target):
    """
    Generate all analysis plots from validation CSVs.
    Returns (paths_dict, stats_dict).
    """
    import numpy as np
    import pandas as pd
    import math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from scipy import stats as sc

    csv_dir   = Path(csv_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True, parents=True)

    required = ['yt_test.csv', 'yh_test.csv', 'yt_train.csv', 'x_test.csv', 'x_train.csv']
    if not all((csv_dir / f).exists() for f in required):
        print(f'  WARNING: some CSVs missing in {csv_dir}')
        return {}, {}

    x_test   = pd.read_csv(csv_dir / 'x_test.csv')
    x_train  = pd.read_csv(csv_dir / 'x_train.csv')
    yt_test  = pd.read_csv(csv_dir / 'yt_test.csv')
    yt_train = pd.read_csv(csv_dir / 'yt_train.csv')
    yh_test  = pd.read_csv(csv_dir / 'yh_test.csv')

    # Optionally load val set
    x_val = yt_val = yh_val = None
    if (csv_dir / 'x_val.csv').exists():
        x_val  = pd.read_csv(csv_dir / 'x_val.csv')
        yt_val = pd.read_csv(csv_dir / 'yt_val.csv')
        yh_val = pd.read_csv(csv_dir / 'yh_val.csv')

    outputs = yt_test.columns.tolist()
    inputs  = x_test.select_dtypes(include='number').columns.tolist()
    n_out   = len(outputs)
    n_inp   = len(inputs)
    ncols   = min(4, n_out)
    nrows   = -(-n_out // ncols)

    residue = yt_test.values - yh_test.values          # (n, n_out)
    abserr  = np.abs(residue)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_abs = abserr / np.where(np.abs(yt_test.values) > 1e-10,
                                    np.abs(yt_test.values), np.nan)

    paths = {}

    def _save(fig, name):
        p = plots_dir / name
        fig.savefig(p, dpi=110, bbox_inches='tight')
        plt.close(fig)
        paths[name.replace('.png', '')] = str(p)
        print(f'  plot: {name}')

    # ── DATA ──────────────────────────────────────────────────────────────────

    # 1. Input statistics table
    x_all = pd.concat([x_train, x_test] + ([x_val] if x_val is not None else []), ignore_index=True)
    stat_rows = []
    for inp in inputs:
        col = x_all[inp].dropna()
        stat_rows.append([inp, f'{len(col):,}', f'{col.mean():.4g}', f'{col.std():.4g}',
                          f'{sc.iqr(col):.4g}', f'{sc.skew(col):.3f}', f'{sc.kurtosis(col):.3f}',
                          f'{col.quantile(.10):.4g}', f'{col.median():.4g}', f'{col.quantile(.90):.4g}'])
    col_labels = ['Input', 'N', 'Mean', 'Std', 'IQR', 'Skew', 'Kurt', 'P10', 'P50', 'P90']
    fig, ax = plt.subplots(figsize=(14, 0.4 * n_inp + 1.2))
    ax.axis('off')
    tbl = ax.table(cellText=stat_rows, colLabels=col_labels, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.5)
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#004680')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    for i in range(1, n_inp + 1):
        bg = '#f2f2f2' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor(bg)
    ax.set_title(f'Input Variable Statistics — {best} (all data)', fontsize=10, pad=8)
    plt.tight_layout()
    _save(fig, 'data_input_stats.png')

    # 2. Input histograms
    n_inp_cols = min(4, n_inp)
    n_inp_rows = -(-n_inp // n_inp_cols)
    fig, axes = plt.subplots(n_inp_rows, n_inp_cols, figsize=(5 * n_inp_cols, 3.5 * n_inp_rows),
                              squeeze=False)
    af = _axes_flat(axes, n_inp_rows, n_inp_cols)
    for i, inp in enumerate(inputs):
        ax = af[i]
        data = x_all[inp].dropna()
        bins = _sturges(len(data))
        ax.hist(data, bins=bins, color='steelblue', alpha=0.75, edgecolor='white', linewidth=0.4)
        ax.set_title(inp, fontsize=9); ax.tick_params(labelsize=7)
    for j in range(n_inp, len(af)): af[j].set_visible(False)
    fig.suptitle(f'Input Histograms — {best}', fontsize=11)
    plt.tight_layout()
    _save(fig, 'data_input_hist.png')

    # 3. Output statistics table
    yt_all = pd.concat([yt_train, yt_test] + ([yt_val] if yt_val is not None else []), ignore_index=True)
    stat_rows_out = []
    for o in outputs:
        col = yt_all[o].dropna()
        stat_rows_out.append([o, f'{len(col):,}', f'{col.mean():.4g}', f'{col.std():.4g}',
                               f'{sc.iqr(col):.4g}', f'{sc.skew(col):.3f}', f'{sc.kurtosis(col):.3f}',
                               f'{col.quantile(.10):.4g}', f'{col.median():.4g}', f'{col.quantile(.90):.4g}'])
    fig, ax = plt.subplots(figsize=(14, 0.4 * n_out + 1.2))
    ax.axis('off')
    tbl = ax.table(cellText=stat_rows_out, colLabels=['Output','N','Mean','Std','IQR','Skew','Kurt','P10','P50','P90'],
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.5)
    for j in range(10):
        tbl[(0, j)].set_facecolor('#004680')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    for i in range(1, n_out + 1):
        bg = '#f2f2f2' if i % 2 == 0 else 'white'
        for j in range(10):
            tbl[(i, j)].set_facecolor(bg)
    ax.set_title(f'Output Variable Statistics — {best} (all data)', fontsize=10, pad=8)
    plt.tight_layout()
    _save(fig, 'data_output_stats.png')

    # 4. Output histograms
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    af = _axes_flat(axes, nrows, ncols)
    for i, o in enumerate(outputs):
        ax = af[i]
        data = yt_all[o].dropna()
        ax.hist(data, bins=_sturges(len(data)), color='steelblue', alpha=0.75,
                edgecolor='white', linewidth=0.4)
        ax.set_title(o, fontsize=9); ax.tick_params(labelsize=7)
    for j in range(n_out, len(af)): af[j].set_visible(False)
    fig.suptitle(f'Output Histograms — {best}', fontsize=11)
    plt.tight_layout()
    _save(fig, 'data_output_hist.png')

    # 5. Output CDFs
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    af = _axes_flat(axes, nrows, ncols)
    for i, o in enumerate(outputs):
        ax = af[i]
        data = np.sort(yt_all[o].dropna().values)
        cdf  = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, cdf, color='steelblue', lw=1.5)
        for q, col in [(0.10, 'orange'), (0.50, 'gray'), (0.90, 'red')]:
            v = np.quantile(data, q)
            ax.axvline(v, lw=0.9, ls=':', color=col, label=f'P{int(q*100)}={v:.3g}')
        ax.set_title(o, fontsize=9); ax.set_ylabel('CDF', fontsize=7)
        ax.legend(fontsize=6); ax.tick_params(labelsize=7)
    for j in range(n_out, len(af)): af[j].set_visible(False)
    fig.suptitle(f'Output CDFs — {best}', fontsize=11)
    plt.tight_layout()
    _save(fig, 'data_output_cdf.png')

    # 5b. Variable correlation scatter — grid of scatter plots (input vs output).
    # One row per input makes this very tall whenever there are more inputs than
    # outputs (7x2 for UCHardLanding, 20x3 for UCAirfoils). Capping the height in
    # LaTeX alone would shrink it to an unreadable stamp, so split the rows into
    # page-sized chunks and emit one figure per chunk instead.
    rows_per_fig = max(3, min(n_inp, int(1.3 * (2.8 * n_out) / 2.2)))
    chunks = [inputs[i:i + rows_per_fig] for i in range(0, n_inp, rows_per_fig)]
    for ci, chunk in enumerate(chunks):
        nrows_c = len(chunk)
        fig, axes = plt.subplots(nrows_c, n_out,
                                  figsize=(2.8 * n_out, 2.2 * nrows_c), squeeze=False)
        for j, inp in enumerate(chunk):
            for k, o in enumerate(outputs):
                ax = axes[j][k]
                xi = x_all[inp].values
                yo = yt_all[o].values
                valid = np.isfinite(xi) & np.isfinite(yo)
                if valid.sum() > 5:
                    ax.scatter(xi[valid], yo[valid], s=1, alpha=0.15,
                               color='steelblue', rasterized=True)
                    r, pv = sc.pearsonr(xi[valid], yo[valid])
                    col = 'red' if abs(r) >= 0.5 else ('darkorange' if abs(r) >= 0.25 else 'gray')
                    ax.set_title(f'r={r:.2f}', fontsize=6, color=col, pad=2)
                if j == nrows_c - 1:
                    ax.set_xlabel(o[:14], fontsize=6)
                if k == 0:
                    ax.set_ylabel(inp[:14], fontsize=6)
                ax.tick_params(labelsize=5)
        part = f'  ({ci + 1}/{len(chunks)})' if len(chunks) > 1 else ''
        fig.suptitle('Variable Correlation — Input vs Output (all data)' + part + '\n'
                     'Red |r|≥0.5   Orange |r|≥0.25   Gray |r|<0.25', fontsize=10)
        plt.tight_layout()
        _save(fig, 'data_scatter_vars.png' if ci == 0
                   else f'data_scatter_vars_{ci + 1}.png')

    # ── TRAIN-TEST SPLIT ──────────────────────────────────────────────────────

    # 6. Double histograms outputs
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    af = _axes_flat(axes, nrows, ncols)
    ks_results = {}
    for i, o in enumerate(outputs):
        ax = af[i]
        tr = yt_train[o].dropna().values
        te = yt_test[o].dropna().values
        lo, hi = min(tr.min(), te.min()), max(tr.max(), te.max())
        bins = np.linspace(lo, hi, _sturges(len(te)))
        ax.hist(tr, bins=bins, density=True, alpha=0.55, label='Train', color='steelblue')
        ax.hist(te, bins=bins, density=True, alpha=0.55, label='Test',  color='tomato')
        ks_stat, ks_pval = sc.ks_2samp(tr, te)
        ks_results[o] = {'ks': ks_stat, 'ks_pval': ks_pval}
        ax.set_title(f'{o}\nKS p={ks_pval:.3f}', fontsize=9,
                     color='green' if ks_pval >= 0.05 else 'red')
        ax.legend(fontsize=7); ax.tick_params(labelsize=7)
    for j in range(n_out, len(af)): af[j].set_visible(False)
    fig.suptitle(f'Train vs Test Output Distributions — {best}', fontsize=11)
    plt.tight_layout()
    _save(fig, 'split_output_dists.png')

    # 7. Double histograms inputs
    fig, axes = plt.subplots(n_inp_rows, n_inp_cols,
                              figsize=(5 * n_inp_cols, 3.5 * n_inp_rows), squeeze=False)
    af = _axes_flat(axes, n_inp_rows, n_inp_cols)
    ks_inputs = {}
    for i, inp in enumerate(inputs):
        ax = af[i]
        tr = x_train[inp].dropna().values
        te = x_test[inp].dropna().values
        lo, hi = min(tr.min(), te.min()), max(tr.max(), te.max())
        bins = np.linspace(lo, hi, _sturges(len(te)))
        ax.hist(tr, bins=bins, density=True, alpha=0.55, label='Train', color='steelblue')
        ax.hist(te, bins=bins, density=True, alpha=0.55, label='Test',  color='tomato')
        _, ks_pval = sc.ks_2samp(tr, te)
        ks_inputs[inp] = ks_pval
        ax.set_title(f'{inp}\nKS p={ks_pval:.3f}', fontsize=8,
                     color='green' if ks_pval >= 0.05 else 'red')
        ax.legend(fontsize=6); ax.tick_params(labelsize=7)
    for j in range(n_inp, len(af)): af[j].set_visible(False)
    fig.suptitle(f'Train vs Test Input Distributions — {best}', fontsize=11)
    plt.tight_layout()
    _save(fig, 'split_input_dists.png')

    # 8. KS / AD table (outputs)
    ks_ad_rows = []
    for o in outputs:
        r  = ks_results.get(o, {})
        tr = yt_train[o].dropna().values
        te = yt_test[o].dropna().values
        try:
            ad_res = sc.anderson_ksamp([tr, te])
            ad_pval = ad_res.significance_level / 100.0
        except Exception:
            ad_pval = float('nan')
        ks_pval = r.get('ks_pval', float('nan'))
        ks_ad_rows.append([o, f'{ks_pval:.4f}',
                           'Pass' if ks_pval >= 0.05 else 'Fail',
                           f'{ad_pval:.4f}' if not math.isnan(ad_pval) else '—',
                           'Pass' if (not math.isnan(ad_pval)) and ad_pval >= 0.05 else 'Fail'])
    fig, ax = plt.subplots(figsize=(10, 0.4 * n_out + 1.2))
    ax.axis('off')
    col_labels = ['Output', 'KS p-val', 'KS', 'AD p-val', 'AD']
    tbl = ax.table(cellText=ks_ad_rows, colLabels=col_labels, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.5)
    for j in range(5):
        tbl[(0, j)].set_facecolor('#004680')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    for i, row in enumerate(ks_ad_rows, 1):
        for j, val in enumerate(row):
            c = 'white'
            if val == 'Pass': c = '#c6efce'
            elif val == 'Fail': c = '#ffc7ce'
            elif j % 2 == 0: c = '#f2f2f2'
            tbl[(i, j)].set_facecolor(c)
    ax.set_title(f'KS and AD Tests: Train vs Test — {best}', fontsize=10, pad=8)
    plt.tight_layout()
    _save(fig, 'split_ks_ad_table.png')

    # ── ERROR QUANTIFICATION ──────────────────────────────────────────────────

    # 9. Residue statistics table
    res_stat_rows = []
    for i, o in enumerate(outputs):
        res = residue[:, i]
        res_stat_rows.append([o, f'{res.mean():.4g}', f'{res.std():.4g}',
                               f'{sc.iqr(res):.4g}', f'{sc.skew(res):.3f}', f'{sc.kurtosis(res):.3f}',
                               f'{np.percentile(res,10):.4g}', f'{np.percentile(res,50):.4g}',
                               f'{np.percentile(res,90):.4g}'])
    fig, ax = plt.subplots(figsize=(14, 0.4 * n_out + 1.2))
    ax.axis('off')
    tbl = ax.table(cellText=res_stat_rows,
                   colLabels=['Output','Mean','Std','IQR','Skew','Kurt','P10','P50','P90'],
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.5)
    for j in range(9):
        tbl[(0, j)].set_facecolor('#004680')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    for i in range(1, n_out + 1):
        # Colour mean cell by whether it's close to 0
        try:
            mean_v = float(res_stat_rows[i-1][1])
            std_v  = float(res_stat_rows[i-1][2])
            ratio  = abs(mean_v) / (std_v + 1e-12)
            tbl[(i, 1)].set_facecolor('#c6efce' if ratio < 0.1 else '#fff2cc' if ratio < 0.3 else '#ffc7ce')
        except Exception:
            pass
        for j in [0] + list(range(2, 9)):
            tbl[(i, j)].set_facecolor('#f2f2f2' if i % 2 == 0 else 'white')
    ax.set_title(f'Residue Statistics (y − ŷ) — {best} (test set)', fontsize=10, pad=8)
    plt.tight_layout()
    _save(fig, 'err_residue_stats.png')

    # 10. Residue histograms with Gaussian fit
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    af = _axes_flat(axes, nrows, ncols)
    for i, o in enumerate(outputs):
        ax = af[i]
        res = residue[:, i]
        bins = _sturges(len(res))
        ax.hist(res, bins=bins, density=True, color='steelblue', alpha=0.65,
                edgecolor='white', linewidth=0.4, label='Residue')
        mu, sig = res.mean(), res.std()
        xs = np.linspace(res.min(), res.max(), 300)
        ax.plot(xs, sc.norm.pdf(xs, mu, sig), 'r-', lw=1.5, label=f'N({mu:.3g},{sig:.3g})')
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.set_title(o, fontsize=9); ax.legend(fontsize=6); ax.tick_params(labelsize=7)
    for j in range(n_out, len(af)): af[j].set_visible(False)
    fig.suptitle(f'Residue Histograms with Gaussian Fit — {best}', fontsize=11)
    plt.tight_layout()
    _save(fig, 'err_residue_hist.png')

    # 11. Residue CDFs
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    af = _axes_flat(axes, nrows, ncols)
    for i, o in enumerate(outputs):
        ax = af[i]
        res = np.sort(residue[:, i])
        cdf = np.arange(1, len(res) + 1) / len(res)
        ax.plot(res, cdf, color='steelblue', lw=1.5)
        ax.axvline(0, color='k', lw=0.8, ls='--')
        for q, col in [(10, 'orange'), (90, 'red')]:
            v = np.percentile(res, q)
            ax.axvline(v, color=col, lw=1, ls=':', label=f'P{q}={v:.3g}')
        ax.set_title(o, fontsize=9); ax.set_ylabel('CDF', fontsize=7)
        ax.legend(fontsize=6); ax.tick_params(labelsize=7)
    for j in range(n_out, len(af)): af[j].set_visible(False)
    fig.suptitle(f'Residue CDF (y − ŷ) — {best}', fontsize=11)
    plt.tight_layout()
    _save(fig, 'err_residue_cdf.png')

    # 12. Abs error histograms
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    af = _axes_flat(axes, nrows, ncols)
    for i, o in enumerate(outputs):
        ax = af[i]
        ae = abserr[:, i]
        ax.hist(ae, bins=_sturges(len(ae)), color='tomato', alpha=0.65,
                edgecolor='white', linewidth=0.4)
        ax.set_title(o, fontsize=9); ax.set_xlabel('|y − ŷ|', fontsize=7)
        ax.tick_params(labelsize=7)
    for j in range(n_out, len(af)): af[j].set_visible(False)
    fig.suptitle(f'Absolute Error Histograms — {best}', fontsize=11)
    plt.tight_layout()
    _save(fig, 'err_abserr_hist.png')

    # 13. Relative abs error CDF + Q90
    q90_results = {}
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    af = _axes_flat(axes, nrows, ncols)
    for i, o in enumerate(outputs):
        ax = af[i]
        re = rel_abs[:, i]
        re = re[~np.isnan(re)]
        re_s = np.sort(re)
        cdf  = np.arange(1, len(re_s) + 1) / len(re_s)
        q90v = float(np.quantile(re, 0.90)) if len(re) > 0 else float('nan')
        q90_results[o] = q90v
        passed = (not math.isnan(q90v)) and q90v < q90_target
        ax.plot(re_s, cdf, color='steelblue', lw=1.5)
        ax.axhline(0.90, color='gray', lw=0.8, ls='--')
        ax.axvline(q90_target, color='red', lw=1.2, ls='--', label=f'Target {q90_target:.0%}')
        if not math.isnan(q90v):
            ax.axvline(q90v, color='orange', lw=1.2, ls=':', label=f'Q90={q90v:.4f}')
        xlim = min(max(q90v * 2 if not math.isnan(q90v) else 0.5, q90_target * 2), 1.0)
        ax.set_xlim(0, xlim)
        ax.set_title(f'{o}\nQ90={q90v:.4f} {"✓" if passed else "✗"}', fontsize=9,
                     color='green' if passed else 'red')
        ax.set_xlabel('|Rel. Error|', fontsize=7); ax.set_ylabel('CDF', fontsize=7)
        ax.legend(fontsize=6); ax.tick_params(labelsize=7)
    for j in range(n_out, len(af)): af[j].set_visible(False)
    fig.suptitle(f'Relative Absolute Error CDF — {best} (target Q90<{q90_target:.0%})', fontsize=11)
    plt.tight_layout()
    _save(fig, 'err_relerr_cdf.png')

    # 14. True vs Predicted distribution comparison (double histogram)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    af = _axes_flat(axes, nrows, ncols)
    for i, o in enumerate(outputs):
        ax = af[i]
        tr = yt_test[o].values
        pr = yh_test[o].values
        lo, hi = min(tr.min(), pr.min()), max(tr.max(), pr.max())
        bins = np.linspace(lo, hi, _sturges(len(tr)))
        ax.hist(tr, bins=bins, density=True, alpha=0.55, label='True', color='steelblue')
        ax.hist(pr, bins=bins, density=True, alpha=0.55, label='Pred', color='tomato')
        _, ks_pval = sc.ks_2samp(tr, pr)
        ax.set_title(f'{o}\nKS p={ks_pval:.3f}', fontsize=9,
                     color='green' if ks_pval >= 0.05 else 'red')
        ax.legend(fontsize=7); ax.tick_params(labelsize=7)
    for j in range(n_out, len(af)): af[j].set_visible(False)
    fig.suptitle(f'True vs Predicted Distribution (test set) — {best}', fontsize=11)
    plt.tight_layout()
    _save(fig, 'err_true_vs_pred_dist.png')

    # ── P(E|X): violin plots per output ───────────────────────────────────────

    # 15. Bias heatmap (KW test)
    kw_pvals = np.ones((n_inp, n_out))
    for j, inp in enumerate(inputs):
        try:
            binned = pd.cut(x_test[inp], bins=_sturges(len(x_test)))
            cats = [c for c in binned.cat.categories if (binned == c).sum() >= 5]
            if len(cats) < 2: continue
            for k in range(n_out):
                groups = [residue[(binned == c).values, k] for c in cats]
                _, pval = sc.kruskal(*groups)
                kw_pvals[j, k] = pval
        except Exception:
            pass
    fig, ax = plt.subplots(figsize=(max(6, n_out * 1.3), max(4, n_inp * 0.6 + 1.5)))
    im = ax.imshow(kw_pvals, aspect='auto', vmin=0, vmax=1, cmap='RdYlGn')
    ax.set_xticks(range(n_out))
    ax.set_xticklabels([o[:14] for o in outputs], rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(n_inp))
    ax.set_yticklabels(inputs, fontsize=8)
    for j in range(n_inp):
        for k in range(n_out):
            c = 'white' if kw_pvals[j, k] < 0.25 else 'black'
            ax.text(k, j, f'{kw_pvals[j,k]:.2f}', ha='center', va='center', fontsize=7, color=c)
    plt.colorbar(im, ax=ax, label='KW p-value (green=no bias, red=bias)')
    ax.set_title(f'Bias Detection: KW p-values (Residue) — {best}\n'
                 f'Rows=inputs, Cols=outputs. p<0.05 → bias', fontsize=9)
    plt.tight_layout()
    _save(fig, 'pex_bias_heatmap.png')

    # 16. P(E|X) violin plots: one figure per output (inputs on x-axis)
    in_ncols = min(4, n_inp)
    in_nrows = -(-n_inp // in_ncols)
    for k, o in enumerate(outputs):
        fig, axes = plt.subplots(in_nrows, in_ncols,
                                  figsize=(4 * in_ncols, 3.5 * in_nrows), squeeze=False)
        af = _axes_flat(axes, in_nrows, in_ncols)
        for j, inp in enumerate(inputs):
            ax = af[j]
            try:
                n_bins = _sturges(len(x_test))
                binned = pd.cut(x_test[inp], bins=n_bins)
                cats = [c for c in binned.cat.categories if (binned == c).sum() >= 3]
                groups = [residue[(binned == c).values, k] for c in cats]
                if groups:
                    parts = ax.violinplot(groups, positions=range(len(groups)), showmedians=True)
                    for pc in parts.get('bodies', []):
                        pc.set_alpha(0.6)
                    mid_idx = len(cats) // 2
                    cat = cats[mid_idx]
                    ax.set_xticks([0, len(cats) // 2, len(cats) - 1])
                    ax.set_xticklabels([f'{cats[0].left:.2g}', f'{cat.mid:.2g}',
                                        f'{cats[-1].right:.2g}'], fontsize=6, rotation=20)
            except Exception:
                pass
            ax.axhline(0, color='k', lw=0.8, ls='--')
            ax.set_title(inp[:15], fontsize=8)
            ax.set_ylabel('Residue' if j % in_ncols == 0 else '', fontsize=7)
            ax.tick_params(labelsize=6)
        for j in range(n_inp, len(af)): af[j].set_visible(False)
        fig.suptitle(f'P(E|X) Residue Violin — Output: {o}', fontsize=10)
        plt.tight_layout()
        safe = o.replace(' ', '_').replace('/', '_')
        _save(fig, f'pex_violin_res_{safe}.png')

    # 17. P(E|X) violin plots for absolute error (same structure)
    for k, o in enumerate(outputs):
        fig, axes = plt.subplots(in_nrows, in_ncols,
                                  figsize=(4 * in_ncols, 3.5 * in_nrows), squeeze=False)
        af = _axes_flat(axes, in_nrows, in_ncols)
        for j, inp in enumerate(inputs):
            ax = af[j]
            try:
                n_bins = _sturges(len(x_test))
                binned = pd.cut(x_test[inp], bins=n_bins)
                cats = [c for c in binned.cat.categories if (binned == c).sum() >= 3]
                groups = [abserr[(binned == c).values, k] for c in cats]
                if groups:
                    parts = ax.violinplot(groups, positions=range(len(groups)), showmedians=True)
                    for pc in parts.get('bodies', []):
                        pc.set_alpha(0.6); pc.set_facecolor('tomato')
                    ax.set_xticks([0, len(cats) // 2, len(cats) - 1])
                    ax.set_xticklabels([f'{cats[0].left:.2g}', f'{cats[len(cats)//2].mid:.2g}',
                                        f'{cats[-1].right:.2g}'], fontsize=6, rotation=20)
            except Exception:
                pass
            ax.set_title(inp[:15], fontsize=8)
            ax.set_ylabel('|Error|' if j % in_ncols == 0 else '', fontsize=7)
            ax.tick_params(labelsize=6)
        for j in range(n_inp, len(af)): af[j].set_visible(False)
        fig.suptitle(f'P(E|X) AbsErr Violin — Output: {o}', fontsize=10)
        plt.tight_layout()
        safe = o.replace(' ', '_').replace('/', '_')
        _save(fig, f'pex_violin_abs_{safe}.png')

    # ── P(E|Y) ────────────────────────────────────────────────────────────────

    # 18. Scatter: residue vs true output
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    af = _axes_flat(axes, nrows, ncols)
    for i, o in enumerate(outputs):
        ax = af[i]
        yt  = yt_test[o].values
        res = residue[:, i]
        ax.scatter(yt, res, s=3, alpha=0.3, color='steelblue', rasterized=True)
        ax.axhline(0, color='k', lw=0.8, ls='--')
        try:
            r, pv = sc.pearsonr(yt, res)
            ax.set_title(f'{o}\nr={r:.3f} p={pv:.3f}', fontsize=9,
                         color='red' if pv < 0.05 else 'black')
        except Exception:
            ax.set_title(o, fontsize=9)
        ax.set_xlabel('True y', fontsize=7); ax.set_ylabel('Residue', fontsize=7)
        ax.tick_params(labelsize=7)
    for j in range(n_out, len(af)): af[j].set_visible(False)
    fig.suptitle(f'P(E|Y) Residue vs True Output — {best}', fontsize=11)
    plt.tight_layout()
    _save(fig, 'pey_residue_scatter.png')

    # 19. Violin: residue binned by true output
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    af = _axes_flat(axes, nrows, ncols)
    for i, o in enumerate(outputs):
        ax = af[i]
        yt  = pd.Series(yt_test[o].values)
        res = residue[:, i]
        try:
            n_bins = min(_sturges(len(yt)), 8)
            binned = pd.cut(yt, bins=n_bins, duplicates='drop')
            cats   = [c for c in binned.cat.categories if (binned == c).sum() >= 3]
            groups = [res[(binned == c).values] for c in cats]
            if groups:
                parts = ax.violinplot(groups, positions=range(len(groups)), showmedians=True)
                for pc in parts.get('bodies', []):
                    pc.set_alpha(0.6)
                ax.set_xticks(range(len(cats)))
                ax.set_xticklabels([f'Q{k+1}' for k in range(len(cats))], fontsize=7)
        except Exception:
            pass
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_title(o, fontsize=9)
        ax.set_xlabel('Output quantile bins', fontsize=7)
        ax.set_ylabel('Residue', fontsize=7)
        ax.tick_params(labelsize=7)
    for j in range(n_out, len(af)): af[j].set_visible(False)
    fig.suptitle(f'P(E|Y) Residue Violin vs True Output Bins — {best}', fontsize=11)
    plt.tight_layout()
    _save(fig, 'pey_residue_violin.png')

    # 20. Scatter: abs error vs true output
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    af = _axes_flat(axes, nrows, ncols)
    for i, o in enumerate(outputs):
        ax = af[i]
        yt = yt_test[o].values
        ae = abserr[:, i]
        ax.scatter(yt, ae, s=3, alpha=0.3, color='tomato', rasterized=True)
        try:
            r, pv = sc.spearmanr(yt, ae)
            ax.set_title(f'{o}\nSpear r={r:.3f} p={pv:.3f}', fontsize=9,
                         color='red' if pv < 0.05 else 'black')
        except Exception:
            ax.set_title(o, fontsize=9)
        ax.set_xlabel('True y', fontsize=7); ax.set_ylabel('|Error|', fontsize=7)
        ax.tick_params(labelsize=7)
    for j in range(n_out, len(af)): af[j].set_visible(False)
    fig.suptitle(f'P(E|Y) Absolute Error vs True Output — {best}', fontsize=11)
    plt.tight_layout()
    _save(fig, 'pey_abserr_scatter.png')

    # 21. Violin: abs error binned by true output
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    af = _axes_flat(axes, nrows, ncols)
    for i, o in enumerate(outputs):
        ax = af[i]
        yt = pd.Series(yt_test[o].values)
        ae = abserr[:, i]
        try:
            n_bins = min(_sturges(len(yt)), 8)
            binned = pd.cut(yt, bins=n_bins, duplicates='drop')
            cats   = [c for c in binned.cat.categories if (binned == c).sum() >= 3]
            groups = [ae[(binned == c).values] for c in cats]
            if groups:
                parts = ax.violinplot(groups, positions=range(len(groups)), showmedians=True)
                for pc in parts.get('bodies', []):
                    pc.set_alpha(0.6); pc.set_facecolor('tomato')
                ax.set_xticks(range(len(cats)))
                ax.set_xticklabels([f'Q{k+1}' for k in range(len(cats))], fontsize=7)
        except Exception:
            pass
        ax.set_title(o, fontsize=9)
        ax.set_xlabel('Output quantile bins', fontsize=7)
        ax.set_ylabel('|Error|', fontsize=7)
        ax.tick_params(labelsize=7)
    for j in range(n_out, len(af)): af[j].set_visible(False)
    fig.suptitle(f'P(E|Y) AbsErr Violin vs True Output Bins — {best}', fontsize=11)
    plt.tight_layout()
    _save(fig, 'pey_abserr_violin.png')

    # ── UNCERTAINTY ───────────────────────────────────────────────────────────

    # 22. Sigma coverage table
    cov_rows = []
    for i, o in enumerate(outputs):
        res   = residue[:, i]
        sigma = res.std()
        c1 = np.mean(np.abs(res) < 1 * sigma)
        c2 = np.mean(np.abs(res) < 2 * sigma)
        c3 = np.mean(np.abs(res) < 3 * sigma)
        cov_rows.append([o, f'{sigma:.4g}', f'{c1:.1%}', f'{c2:.1%}', f'{c3:.1%}',
                         f'{abs(c1-0.683):.3f}', f'{abs(c2-0.954):.3f}', f'{abs(c3-0.997):.3f}'])
    fig, ax = plt.subplots(figsize=(13, 0.4 * n_out + 1.2))
    ax.axis('off')
    col_labels = ['Output', 'σ', 'Cov ±1σ', 'Cov ±2σ', 'Cov ±3σ',
                  'Δ from 68.3%', 'Δ from 95.4%', 'Δ from 99.7%']
    tbl = ax.table(cellText=cov_rows, colLabels=col_labels, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.5)
    expected = [0.683, 0.954, 0.997]
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#004680')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')
    for i, row in enumerate(cov_rows, 1):
        for j_c, (c_idx, exp) in enumerate([(2, 0.683), (3, 0.954), (4, 0.997)]):
            val = float(row[c_idx].strip('%')) / 100
            diff = abs(val - exp)
            color = '#c6efce' if diff < 0.03 else '#fff2cc' if diff < 0.10 else '#ffc7ce'
            tbl[(i, c_idx)].set_facecolor(color)
        for j in [0, 1, 5, 6, 7]:
            tbl[(i, j)].set_facecolor('#f2f2f2' if i % 2 == 0 else 'white')
    ax.set_title(f'Uncertainty — Sigma Coverage — {best}\n'
                 f'Expected for Gaussian: ±1σ=68.3%, ±2σ=95.4%, ±3σ=99.7%\n'
                 f'Green=within 3%, Yellow=within 10%, Red=deviates significantly', fontsize=9)
    plt.tight_layout()
    _save(fig, 'uncertainty_coverage.png')

    # 23. Uncertainty bounds scatter (predicted + ±2σ ribbon per output)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    af = _axes_flat(axes, nrows, ncols)
    for i, o in enumerate(outputs):
        ax = af[i]
        yt_v  = yt_test[o].values
        yh_v  = yh_test[o].values
        sigma = residue[:, i].std()
        idx   = np.argsort(yt_v)
        ax.scatter(yt_v[idx], yh_v[idx], s=3, alpha=0.3, color='steelblue', label='Pred')
        ax.fill_between(yt_v[idx], yh_v[idx] - 2*sigma, yh_v[idx] + 2*sigma,
                        alpha=0.15, color='orange', label='±2σ')
        lo = min(yt_v.min(), yh_v.min())
        hi = max(yt_v.max(), yh_v.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        ax.set_title(o, fontsize=9)
        ax.set_xlabel('True', fontsize=7); ax.set_ylabel('Predicted', fontsize=7)
        ax.legend(fontsize=6); ax.tick_params(labelsize=7)
    for j in range(n_out, len(af)): af[j].set_visible(False)
    fig.suptitle(f'Uncertainty Bounds (±2σ) — {best}', fontsize=11)
    plt.tight_layout()
    _save(fig, 'uncertainty_bounds.png')

    # Stats summary for prose
    res_stats_dict = {o: {
        'mean': float(residue[:, i].mean()),
        'std':  float(residue[:, i].std()),
        'p10':  float(np.percentile(residue[:, i], 10)),
        'p90':  float(np.percentile(residue[:, i], 90)),
    } for i, o in enumerate(outputs)}

    stats = dict(
        q90_results=q90_results,
        res_stats=res_stats_dict,
        kw_pvals=kw_pvals,
        inputs=inputs,
        outputs=outputs,
        ks_results=ks_results,
        cov_rows=cov_rows,
    )
    return paths, stats


# ── Part 3 LaTeX ───────────────────────────────────────────────────────────────

def _part3(d, best, paths, stats, out_dir):
    uc_esc   = _esc(d['use_case'])
    best_esc = _esc(best)
    n_out    = len(d['outputs'])
    outputs  = d['outputs']
    q90_target = d['q90_target']

    if not paths:
        return (
            '% ==== PART 3 - DEEP ANALYSIS (no CSVs) ====\n'
            r'\label{sec:deepanalysis}' + '\n'
            rf'\section*{{Deep Analysis --- {best_esc}}}' + '\n'
            r'\noindent\textit{Validation CSVs not found. Run the pipeline first.}' + '\n'
        )

    def _rp(name):
        p = paths.get(name, '')
        if not p: return ''
        return os.path.relpath(p, str(out_dir))

    def _fig(name, width='1.0'):
        rp = _rp(name)
        if not rp: return ''
        return (r'\begin{center}\includegraphics[width=' + width +
                r'\linewidth,' + _GFX_MAXH + r',keepaspectratio]{'
                + rp + r'}\end{center}' + '\n')

    target_pct = _pct(q90_target)
    q90_res    = stats.get('q90_results', {})
    res_stats  = stats.get('res_stats', {})
    kw_pvals   = stats.get('kw_pvals')
    inputs     = stats.get('inputs', [])

    # Q90 summary table for LaTeX
    q90_tbl_rows = []
    for o in outputs:
        val = q90_res.get(o)
        vs  = f'{val:.4f}' if val is not None else r'\textemdash'
        ok  = val is not None and val < q90_target
        cc  = 'passgreen' if ok else 'failred'
        st  = r'\textbf{PASS}' if ok else r'\textbf{FAIL}'
        q90_tbl_rows.append(
            _esc(o) + ' & '
            + rf'\cellcolor{{{cc}}}{vs}' + ' & '
            + target_pct + ' & '
            + rf'\cellcolor{{{cc}}}{st}'
            + r' \\ \hline'
        )
    q90_tbl_tex = '\n  '.join(q90_tbl_rows)

    # Bias prose
    import numpy as np
    biased = []
    if kw_pvals is not None:
        for j, inp in enumerate(inputs):
            for k, o in enumerate(outputs):
                if kw_pvals[j, k] < 0.05:
                    biased.append(_esc(inp) + r'$\to$' + _esc(o))
    if biased:
        biased_tex = r'\textcolor{red}{\textit{' + ', '.join(biased[:10]) + (r'\ldots' if len(biased) > 10 else '') + r'}}'
        bias_prose = (f'Significant input-conditional bias ($p<0.05$) detected for: {biased_tex}. '
                      r'These inputs explain residual variance --- consider polynomial features or interaction terms.')
    else:
        bias_prose = (r'No significant input-conditional bias detected ($p\geq 0.05$ for all input--output pairs). '
                      r'The residue does not depend systematically on any input.')

    # P(E|X) violin blocks (one per output)
    pex_res_blocks, pex_abs_blocks = '', ''
    for o in outputs:
        safe = o.replace(' ', '_').replace('/', '_')
        fig_r = _fig(f'pex_violin_res_{safe}')
        fig_a = _fig(f'pex_violin_abs_{safe}')
        if fig_r:
            pex_res_blocks += rf'\subsection*{{\normalsize Residue --- {_esc(o)}}}' + '\n' + fig_r
        if fig_a:
            pex_abs_blocks += rf'\subsection*{{\normalsize Absolute Error --- {_esc(o)}}}' + '\n' + fig_a

    mini_toc = (
        r'\begin{itemize}[leftmargin=2em,itemsep=1pt,topsep=2pt]' + '\n'
        r'  \item \hyperref[sec:d1]{3.1 Data Overview}' + '\n'
        r'  \item \hyperref[sec:d2]{3.2 Train-Test Split Analysis}' + '\n'
        r'  \item \hyperref[sec:d3]{3.3 Error Quantification --- P(E)}' + '\n'
        r'  \item \hyperref[sec:d4]{3.4 P(E|X) --- Error Conditional on Inputs}' + '\n'
        r'  \item \hyperref[sec:d5]{3.5 P(E|Y) --- Error Conditional on Outputs}' + '\n'
        r'  \item \hyperref[sec:d6]{3.6 Uncertainty Analysis}' + '\n'
        r'\end{itemize}'
    )

    return (
        '% ==== PART 3 - DEEP ANALYSIS ====\n'
        r'\label{sec:deepanalysis}' + '\n'
        r'\begin{center}' + '\n'
        rf'  {{\normalsize\bfseries\color{{primary}} Deep Analysis --- {best_esc}}}\\[2pt]' + '\n'
        rf'  {{\small\color{{sepgray}} Mirrors \texttt{{validation\_template.ipynb}} --- all computations from validation CSVs}}' + '\n'
        r'\end{center}' + '\n'
        r'\vspace{4pt}' + '\n'
        + mini_toc + '\n\n'

        # 3.1 Data Overview
        r'\clearpage' + '\n'
        r'\section{Data Overview}' + '\n'
        r'\label{sec:d1}' + '\n\n'
        rf'Statistical summary of inputs and outputs for \textbf{{{best_esc}}} (all data: train + val + test combined). '
        r'IQR = interquartile range; Skew and Kurt are the Fisher skewness and excess kurtosis. '
        r'P10/P50/P90 are the 10th, 50th, and 90th percentiles.' + '\n\n'
        r'\exhead{Input Statistics}' + '\n'
        + _fig('data_input_stats') +
        r'\exhead{Input Histograms}' + '\n'
        + _fig('data_input_hist') +
        r'\exhead{Output Statistics}' + '\n'
        + _fig('data_output_stats') +
        r'\exhead{Output Histograms}' + '\n'
        + _fig('data_output_hist') +
        r'\exhead{Output Cumulative Distributions}' + '\n'
        r'Vertical markers show P10, P50, and P90.' + '\n'
        + _fig('data_output_cdf')

        # 3.2 Train-Test Split
        + r'\clearpage' + '\n'
        r'\section{Train-Test Split Analysis}' + '\n'
        r'\label{sec:d2}' + '\n\n'
        r'A well-designed split produces training and test distributions that are similar but not identical. '
        r'The Kolmogorov-Smirnov (KS) test and the Anderson-Darling (AD) test are used to assess distributional similarity. '
        r'$H_0$: both sets come from the same distribution. '
        r'$p<0.05$ rejects $H_0$ (undesirable if too low; undesirable if the split was done by stratification and $p$ is too high).' + '\n\n'
        r'\exhead{Output Distributions: Train vs Test}' + '\n'
        + _fig('split_output_dists') +
        r'\exhead{Input Distributions: Train vs Test}' + '\n'
        + _fig('split_input_dists') +
        r'\exhead{KS and AD Test Results}' + '\n'
        + _fig('split_ks_ad_table')

        # 3.3 Error Quantification
        + r'\clearpage' + '\n'
        r'\section{Error Quantification --- P(E)}' + '\n'
        r'\label{sec:d3}' + '\n\n'
        r'Two error metrics are studied:' + '\n'
        r'\begin{itemize}[leftmargin=1.4em,itemsep=1pt,topsep=1pt]' + '\n'
        r'  \item \textbf{Residue:} $e_i = y_i - \hat{y}_i$. '
        r'A centred ($\mu\approx 0$), symmetric distribution indicates an unbiased model.' + '\n'
        r'  \item \textbf{Absolute Error:} $|e_i|$. '
        r'Always non-negative; its Q90 must satisfy the accuracy requirement.' + '\n'
        r'\end{itemize}' + '\n\n'
        r'\exhead{Residue Statistics Table}' + '\n'
        r'Mean highlighted: green $|\mu|/\sigma<0.10$, yellow $<0.30$, red $\geq 0.30$.' + '\n'
        + _fig('err_residue_stats') +
        r'\exhead{Residue Histograms with Gaussian Fit}' + '\n'
        r'Red curve = fitted normal distribution. Deviation from the normal indicates heavy tails or skew.' + '\n'
        + _fig('err_residue_hist') +
        r'\exhead{Residue CDF}' + '\n'
        + _fig('err_residue_cdf') +
        r'\exhead{Absolute Error Histograms}' + '\n'
        + _fig('err_abserr_hist') +
        r'\exhead{Relative Absolute Error CDF --- Q90 Accuracy Requirement}' + '\n'
        r'Orange dotted = observed Q90; red dashed = target threshold. '
        r'Green title = PASS; red title = FAIL.' + '\n'
        + _fig('err_relerr_cdf')
        + _tbl_with_legend(
            r'\renewcommand{\arraystretch}{1.2}' + '\n'
            r'\begin{tabular}{|l|r|r|c|}' + '\n'
            r'  \hline\textbf{Output} & \textbf{Q90} & \textbf{Target} & \textbf{Status} \\ \hline' + '\n'
            f'  {q90_tbl_tex}\n'
            r'\end{tabular}',
            _KS_LEGEND.replace('no overfitting', 'pass').replace('overfitting detected', 'fail')
        ) + '\n\n'
        r'\exhead{True vs Predicted Distribution Comparison}' + '\n'
        r'KS test comparing the distribution of true vs.\ predicted values on the test set. '
        r'A significant difference indicates the model is not reproducing the output distribution faithfully.' + '\n'
        + _fig('err_true_vs_pred_dist')

        # 3.4 P(E|X)
        + r'\clearpage' + '\n'
        r'\section{P(E|X) --- Error Conditional on Inputs}' + '\n'
        r'\label{sec:d4}' + '\n\n'
        r'We study whether the model error depends on the input variables. '
        r'Two approaches are used: (1) a Kruskal-Wallis (KW) test on binned inputs '
        r'(Sturges rule for bin count), and (2) violin plots showing the error distribution '
        r'within each input bin.' + '\n\n'
        + bias_prose + '\n\n'
        r'\exhead{Bias Detection Heatmap (KW p-values)}' + '\n'
        r'Rows = inputs, columns = outputs. '
        r'Green ($p\geq 0.05$) = residue uniform across input bins; '
        r'red ($p<0.05$) = significant bias.' + '\n'
        + _fig('pex_bias_heatmap')
        + r'\clearpage' + '\n'
        r'\exhead{Residue Violin Plots per Output (binned by input)}' + '\n'
        r'Each panel shows the residue distribution for each input bin. '
        r'A flat median line (dashed at 0) and symmetric violins indicate absence of bias.' + '\n\n'
        + pex_res_blocks
        + r'\clearpage' + '\n'
        r'\exhead{Absolute Error Violin Plots per Output (binned by input)}' + '\n'
        + pex_abs_blocks

        # 3.5 P(E|Y)
        + r'\clearpage' + '\n'
        r'\section{P(E|Y) --- Error Conditional on Outputs}' + '\n'
        r'\label{sec:d5}' + '\n\n'
        r'We study whether the error depends on the output magnitude. '
        r'A significant trend (Pearson or Spearman $p<0.05$) means the model is more accurate '
        r'in some output ranges than others --- a form of heteroscedasticity.' + '\n\n'
        r'\exhead{Residue vs True Output --- Scatter}' + '\n'
        r'Pearson r and p-value shown in title. Red title = significant linear trend.' + '\n'
        + _fig('pey_residue_scatter') +
        r'\exhead{Residue Violin vs True Output Bins}' + '\n'
        r'Bins determined by Sturges rule. Median line dashed at 0.' + '\n'
        + _fig('pey_residue_violin') +
        r'\exhead{Absolute Error vs True Output --- Scatter}' + '\n'
        r'Spearman r shown. Red title = significant monotonic trend.' + '\n'
        + _fig('pey_abserr_scatter') +
        r'\exhead{Absolute Error Violin vs True Output Bins}' + '\n'
        + _fig('pey_abserr_violin')

        # 3.6 Uncertainty
        + r'\clearpage' + '\n'
        r'\section{Uncertainty Analysis}' + '\n'
        r'\label{sec:d6}' + '\n\n'
        r'A global Gaussian uncertainty model is estimated from the residue standard deviation on the test set. '
        r'Coverage is the empirical proportion of test points within $\pm k\sigma$ bounds. '
        r'For a perfectly Gaussian error, $\pm 1\sigma$ should cover 68.3\%, $\pm 2\sigma$ 95.4\%, '
        r'$\pm 3\sigma$ 99.7\%. '
        r'Green = within 3\% of Gaussian expectation, yellow = within 10\%, red = significant deviation.' + '\n\n'
        r'\exhead{Sigma Coverage Table}' + '\n'
        + _fig('uncertainty_coverage') +
        r'\exhead{Uncertainty Bounds $\pm 2\sigma$ on Predicted vs True}' + '\n'
        r'Orange ribbon = $\pm 2\sigma$ around the prediction. '
        r'Points outside the ribbon are larger errors than expected under the Gaussian model.' + '\n'
        + _fig('uncertainty_bounds')
    )


# ── HTML export ────────────────────────────────────────────────────────────────

def _generate_html(d, best, paths, stats, out_dir):
    uc       = d['use_case']
    date_str = datetime.today().strftime('%d %B %Y')

    def _img(name):
        p = paths.get(name, '')
        if not p or not Path(p).exists(): return ''
        data = base64.b64encode(Path(p).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" style="width:90%;max-width:900px;display:block;margin:12px auto;">'

    q90_res  = stats.get('q90_results', {})
    q90_target = d['q90_target']
    cov_rows   = stats.get('cov_rows', [])
    outputs    = d['outputs']
    inputs_l   = stats.get('inputs', [])

    def _q90_tbl():
        rows = ''
        for o in outputs:
            v  = q90_res.get(o)
            vs = f'{v:.4f}' if v is not None else '—'
            ok = v is not None and v < q90_target
            c  = '#c6efce' if ok else '#ffc7ce'
            rows += f'<tr><td>{o}</td><td style="background:{c}">{vs}</td><td>{q90_target:.0%}</td><td style="background:{c}"><b>{"PASS" if ok else "FAIL"}</b></td></tr>'
        return f'<table><tr><th>Output</th><th>Q90</th><th>Target</th><th>Status</th></tr>{rows}</table>'

    def _cov_tbl():
        if not cov_rows: return ''
        rows = ''.join(
            f'<tr>{"".join(f"<td>{v}</td>" for v in row)}</tr>'
            for row in cov_rows)
        hdrs = '<tr><th>Output</th><th>σ</th><th>Cov±1σ</th><th>Cov±2σ</th><th>Cov±3σ</th><th>Δ68.3%</th><th>Δ95.4%</th><th>Δ99.7%</th></tr>'
        return f'<table>{hdrs}{rows}</table>'

    css = """body{font-family:Arial,sans-serif;font-size:10pt;margin:40px auto;max-width:1100px;color:#222}
h1{color:#004680;border-bottom:2px solid #004680;padding-bottom:6px}
h2{color:#004680;margin-top:36px;border-bottom:1px solid #b0c4de}
h3{color:#555;margin-top:20px}
p{line-height:1.5}
table{border-collapse:collapse;margin:10px 0;font-size:9pt}
th{background:#004680;color:white;padding:5px 10px;text-align:center}
td{padding:4px 10px;border:1px solid #ccc;text-align:center}
tr:nth-child(even){background:#f5f5f5}
.meta{color:#777;font-size:9pt}
nav ul{list-style:none;padding:0}
nav li{margin:4px 0}
nav a{color:#004680}"""

    def _section(anchor, title, content):
        return f'<h2 id="{anchor}">{title}</h2>\n{content}\n'

    pex_res_html = ''
    pex_abs_html = ''
    for o in outputs:
        safe = o.replace(' ', '_').replace('/', '_')
        pex_res_html += f'<h3>Residue — {o}</h3>' + _img(f'pex_violin_res_{safe}')
        pex_abs_html += f'<h3>Absolute Error — {o}</h3>' + _img(f'pex_violin_abs_{safe}')

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>Deep Analysis — {uc}</title>
<style>{css}</style></head><body>
<h1>Deep Analysis &mdash; {best}</h1>
<p class="meta">Use case: <b>{uc}</b> &nbsp;|&nbsp; {date_str} &nbsp;|&nbsp; Mirrors validation_template.ipynb</p>

<nav><b>Contents</b><ul>
<li><a href="#d1">3.1 Data Overview</a></li>
<li><a href="#d2">3.2 Train-Test Split Analysis</a></li>
<li><a href="#d3">3.3 Error Quantification &mdash; P(E)</a></li>
<li><a href="#d4">3.4 P(E|X) &mdash; Error Conditional on Inputs</a></li>
<li><a href="#d5">3.5 P(E|Y) &mdash; Error Conditional on Outputs</a></li>
<li><a href="#d6">3.6 Uncertainty Analysis</a></li>
</ul></nav>

{_section("d1","3.1 Data Overview",
    "<h3>Input Statistics</h3>" + _img("data_input_stats") +
    "<h3>Input Histograms</h3>" + _img("data_input_hist") +
    "<h3>Output Statistics</h3>" + _img("data_output_stats") +
    "<h3>Output Histograms</h3>" + _img("data_output_hist") +
    "<h3>Output CDFs</h3>" + _img("data_output_cdf")
)}

{_section("d2","3.2 Train-Test Split Analysis",
    "<h3>Output Distributions</h3>" + _img("split_output_dists") +
    "<h3>Input Distributions</h3>" + _img("split_input_dists") +
    "<h3>KS &amp; AD Test Table</h3>" + _img("split_ks_ad_table")
)}

{_section("d3","3.3 Error Quantification &mdash; P(E)",
    "<h3>Residue Statistics</h3>" + _img("err_residue_stats") +
    "<h3>Residue Histograms (Gaussian fit)</h3>" + _img("err_residue_hist") +
    "<h3>Residue CDFs</h3>" + _img("err_residue_cdf") +
    "<h3>Absolute Error Histograms</h3>" + _img("err_abserr_hist") +
    "<h3>Relative Error CDFs — Q90 Requirement</h3>" + _img("err_relerr_cdf") + _q90_tbl() +
    "<h3>True vs Predicted Distributions</h3>" + _img("err_true_vs_pred_dist")
)}

{_section("d4","3.4 P(E|X) &mdash; Error Conditional on Inputs",
    "<h3>Bias Heatmap (KW p-values)</h3>" + _img("pex_bias_heatmap") +
    "<h3>Residue Violin per Output</h3>" + pex_res_html +
    "<h3>Absolute Error Violin per Output</h3>" + pex_abs_html
)}

{_section("d5","3.5 P(E|Y) &mdash; Error Conditional on Outputs",
    "<h3>Residue vs True Output (scatter)</h3>" + _img("pey_residue_scatter") +
    "<h3>Residue Violin vs Output Bins</h3>" + _img("pey_residue_violin") +
    "<h3>Absolute Error vs True Output (scatter)</h3>" + _img("pey_abserr_scatter") +
    "<h3>Absolute Error Violin vs Output Bins</h3>" + _img("pey_abserr_violin")
)}

{_section("d6","3.6 Uncertainty Analysis",
    "<h3>Sigma Coverage Table</h3>" + _cov_tbl() + _img("uncertainty_coverage") +
    "<h3>±2σ Bounds on Predicted vs True</h3>" + _img("uncertainty_bounds")
)}

<hr><p class="meta">Generated by Surrogate Factory v2.2 &mdash; Automated Validation Report</p>
</body></html>"""

    out_file = Path(out_dir) / f'deep_analysis_{uc}.html'
    out_file.write_text(html, encoding='utf-8')
    print(f'HTML report      → {out_file}')


# ── Build ──────────────────────────────────────────────────────────────────────

def build_latex(d, scatter_paths, paths, stats, out_dir):
    best = d['best_model']
    rows = _per_output(d, best)
    uc   = d['use_case']
    preamble = _PREAMBLE.replace('XUSECASEX', _esc(uc))
    return (
        preamble
        + r'\begin{document}' + '\n'
        + _part1(d, best, rows, scatter_paths, paths, out_dir)
        + _separator(uc)
        + _part3(d, best, paths, stats, out_dir)
        + '\n' + r'\end{document}' + '\n'
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def _build_pdf(tex_file, out_dir):
    """
    Compile the .tex with whichever LaTeX engine is available.

    tectonic is preferred (self-contained, downloads its own packages), but it
    is not installable via pip and not always present. The emitted .tex uses
    only standard packages and pure-ASCII content, so any TeX Live engine can
    build it. Returns the engine name, or None if none succeeded.
    """
    import shutil as _shutil

    tex, odir = str(tex_file), str(out_dir)

    # (executable, argv, passes) — pdflatex/xelatex need two passes to
    # resolve the table of contents and hyperref references.
    candidates = [
        ('tectonic', ['tectonic', '-o', odir, tex], 1),
        ('latexmk',  ['latexmk', '-pdf', '-interaction=nonstopmode',
                      '-halt-on-error', f'-outdir={odir}', tex], 1),
        ('xelatex',  ['xelatex', '-interaction=nonstopmode', '-halt-on-error',
                      '-output-directory', odir, tex], 2),
        ('pdflatex', ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                      '-output-directory', odir, tex], 2),
    ]

    tried, last_log = [], ''
    for exe, argv, passes in candidates:
        if _shutil.which(exe) is None:
            tried.append(f'{exe}: not installed')
            continue
        for _ in range(passes):
            # cwd=out_dir so the relative image paths in the .tex resolve.
            result = subprocess.run(argv, capture_output=True, text=True, cwd=odir)
        if result.returncode == 0:
            return exe
        tried.append(f'{exe}: exited {result.returncode}')
        last_log = (result.stdout or '') + (result.stderr or '')

    print('ERROR: could not build the PDF. Engines tried:')
    for t in tried:
        print(f'  - {t}')
    if last_log:
        print('\nLast engine output:\n' + last_log[-3000:])
    else:
        print('\nNo LaTeX engine found. tectonic is NOT available via pip;')
        print('install the static binary (no root, no conda):')
        print('  mkdir -p ~/.local/bin && cd ~/.local/bin')
        print('  curl --proto "=https" --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh')
        print('  export PATH="$HOME/.local/bin:$PATH"      # add to ~/.bashrc')
        print('\nAlternatively any TeX Live install works (latexmk / xelatex / pdflatex).')
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata', help='Path to metadata_*.json')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    out_dir   = (Path(args.output).resolve()
                 if args.output else meta_path.parent.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)

    d    = extract(meta_path)
    best = d['best_model']
    artifacts_dir = meta_path.parent.resolve()

    scatter_paths = {}
    for m in d['models']:
        png = artifacts_dir / f'scatter_{m}.png'
        if png.exists():
            scatter_paths[m] = os.path.relpath(str(png), str(out_dir))

    csv_dir   = artifacts_dir / f'validation_{best}'
    plots_dir = out_dir / 'analysis_plots'
    print(f'Generating analysis plots from {csv_dir} ...')
    paths, stats = _analysis_plots(best, csv_dir, plots_dir, d['q90_target'])

    use_case = d['use_case']
    tex_file = out_dir / f'executive_summary_{use_case}.tex'
    pdf_file = out_dir / f'executive_summary_{use_case}.pdf'

    tex = build_latex(d, scatter_paths, paths, stats, out_dir)
    tex_file.write_text(tex, encoding='utf-8')
    print(f'LaTeX source     → {tex_file}')

    engine = _build_pdf(tex_file, out_dir)
    if engine:
        print(f'PDF generated    → {pdf_file}   (engine: {engine})')
    else:
        sys.exit(1)

    if paths:
        _generate_html(d, best, paths, stats, out_dir)


if __name__ == '__main__':
    main()
