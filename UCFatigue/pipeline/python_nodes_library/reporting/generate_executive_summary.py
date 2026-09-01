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


# ── Split quality (VTPM) table ─────────────────────────────────────────────────
# Shared by Part 1 and Part 3. The voxel-tesselation proximity method from
# validationlib flags three kinds of test point: p-hacking (too close to a
# training point — the test would flatter the model), isolated (too far from
# any training point — outside what was learned) and residual-voxel (in a
# region with no training data at all). The residual-voxel and chi-squared
# thresholds are the library's own; the 5 % marks on the p-hacking and isolated
# proportions are warning levels, not pass/fail criteria.
PHACK_WARN = 0.05
ISOL_WARN  = 0.05


def _split_quality_tbl(split):
    split = split or {}

    def _sv(key, fmt='.3f'):
        v = split.get(key)
        return f'{v:{fmt}}' if v is not None else r'\textemdash'

    def _si(key, op, th):
        v = split.get(key)
        if v is None: return r'\textemdash'
        ok = (v <= th) if op == '<=' else (v >= th)
        return r'\cellcolor{passgreen}$\checkmark$' if ok else r'\cellcolor{failred}$\times$'

    return (
        r'\begin{minipage}{\linewidth}' + '\n'
        r'\renewcommand{\arraystretch}{1.2}' + '\n'
        r'\begin{tabular}{|l|r|c|}' + '\n'
        r'  \hline\textbf{Metric} & \textbf{Value} & \textbf{Pass} \\ \hline' + '\n'
        rf'  Residual voxel proportion ($\leq\!0.05$) & {_sv("residual_voxel_proportion")} & {_si("residual_voxel_proportion","<=",0.05)} \\ \hline' + '\n'
        rf'  Valid test proportion & {_sv("valid_test_proportion")} & \\ \hline' + '\n'
        rf'  p-hacking test proportion ($\leq\!{PHACK_WARN:.2f}$ warn) & {_sv("phacking_test_proportion")} & {_si("phacking_test_proportion","<=",PHACK_WARN)} \\ \hline' + '\n'
        rf'  Isolated test proportion ($\leq\!{ISOL_WARN:.2f}$ warn) & {_sv("isolated_test_proportion")} & {_si("isolated_test_proportion","<=",ISOL_WARN)} \\ \hline' + '\n'
        rf'  Chi\textsuperscript{{2}} p-value ($\geq\!0.05$) & {_sv("chi_squared_pvalue",".4f")} & {_si("chi_squared_pvalue",">=",0.05)} \\ \hline' + '\n'
        r'\end{tabular}' + '\n'
        r'\end{minipage}'
    )


def _split_quality_prose(split):
    """One-paragraph reading of the VTPM result, for the Train-Test section."""
    split = split or {}
    ph, iso = split.get('phacking_test_proportion'), split.get('isolated_test_proportion')
    if ph is None:
        return (r'\textit{Split validation was not recorded in the metadata '
                r'(SF\_9 cell 9.0); run it to obtain the p-hacking check.}')
    # Two different findings, two different readings: p-hacking makes the test
    # error optimistic, isolation makes it pessimistic (extrapolation).
    if ph <= PHACK_WARN:
        phack = (rf'\textcolor{{passgreen!60!black}}{{\textbf{{No p-hacking concern:}}}} '
                 rf'only {_pct(ph, 1)} of test points lie closer to a training point than '
                 rf'training points are to each other, so the test error is not flattered.')
    else:
        phack = (rf'\textcolor{{failred!70!black}}{{\textbf{{p-hacking risk:}}}} {_pct(ph, 1)} '
                 rf'of test points sit closer to a training point than training points are '
                 rf'to each other. Errors on those points are optimistic; read the reported '
                 rf'accuracy with that in mind, or redraw the split.')
    if iso is None or iso <= ISOL_WARN:
        isol = ''
    else:
        isol = (rf' \textcolor{{failred!70!black}}{{\textbf{{Isolated points:}}}} {_pct(iso, 1)} '
                rf'of test points have no training point nearby, so part of the test error '
                rf'measures extrapolation rather than interpolation --- the model may look '
                rf'worse there than it is in the region it was trained on.')
    return phack + isol


# ── Data extraction ────────────────────────────────────────────────────────────

def _scatter_cfg_from(meta_path: Path, val: dict) -> dict:
    """
    Which variables the correlation matrix shows.

    The SF_9 yaml next to the pipeline is the source of truth: the metadata
    JSON only receives a copy of it when SF_9 is re-run through
    import_metadata, so an edit to the yaml would otherwise be ignored until
    the whole validation stage is executed again.
    """
    cfg = {'variables': list(val.get('scatter_variables') or []),
           'method': val.get('scatter_method') or 'scatter'}
    yml = meta_path.resolve().parent.parent.parent / 'metadata' / 'SF_9_Model_Validation.yaml'
    if yml.exists():
        try:
            import yaml
            mv = (yaml.safe_load(yml.read_text()) or {}).get('Model_Validation') or {}
            if mv.get('scatter_variables'):
                cfg['variables'] = list(mv['scatter_variables'])
            if mv.get('scatter_method'):
                cfg['method'] = mv['scatter_method']
        except Exception as e:
            print(f'  scatter config: could not read {yml.name} ({type(e).__name__}); using metadata')
    return cfg


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
        # Needed to recover the training curves, which live inside the fitted
        # estimators rather than in the metadata.
        model_files={m['label']: m.get('file') for m in trn.get('Models', [])},
        scatter_cfg=_scatter_cfg_from(meta_path, val),
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
# Two figures get their own box. The training curve is an inset, not a page
# filler; the correlation matrix must sit whole on the page its section opens.
_GFX_CURVE  = r'width=0.70\linewidth,height=0.28\textheight,keepaspectratio'
_GFX_MATRIX = r'width=0.80\linewidth,height=0.46\textheight,keepaspectratio'


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

def _part1(d, best, rows, scatter_paths, paths, out_dir, stats=None):
    tables = (stats or {}).get('tables') or {}

    def _tbl(key):
        """A native LaTeX table produced alongside the validationlib figures."""
        return tables.get(key, {}).get('tex', '')

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

    def _training_curve_tex():
        """Training history section — omitted when no model recorded one."""
        p = paths.get('training_curve', '')
        if not p:
            return ''
        rp = os.path.relpath(p, str(out_dir))
        return (
            r'\section{Training History}' + '\n'
            + r'\begin{center}\includegraphics[' + _GFX_CURVE + r']{'
            + rp + r'}\end{center}' + '\n'
            + r'{\footnotesize Training loss per iteration, log scale. '
              r'A validation panel is shown for models trained with early '
              r'stopping. Gradient boosting reports per-stage deviance '
              r'averaged over its per-output estimators.}' + '\n\n'
        )

    # ── Winner scatter ─────────────────────────────────────────────────────────
    # Prefer the validationlib predicted-vs-true figure (same one the extended
    # validation notebook draws); the SF_9 scatter_<model>.png is the fallback.
    winner_scatter_tex = _fig_es('winner_pred_true')
    if not winner_scatter_tex:
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
    # Three columns per model: with five models (UCLoads) that is 16 columns and
    # the bare tabular runs off the page, clipping whole model blocks. Scale it
    # to the line width whenever more than two models are present.
    metrics_inner = (
        r'\renewcommand{\arraystretch}{1.2}' + '\n'
        rf'\begin{{tabular}}{{{col_spec}|}}' + '\n'
        r'  \hline' + '\n'
        rf'  \textbf{{Output}} & {mdl_span} \\ \hline' + '\n'
        rf'  & {sub_hdr[3:]} \\ \hline' + '\n'
        f'  {metrics_tex}\n'
        r'\end{tabular}'
    )
    if len(models) > 2:
        metrics_tbl = (r'\resizebox{\linewidth}{!}{%' + '\n'
                       + metrics_inner + '\n' + r'}')
    else:
        metrics_tbl = metrics_inner

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

    # ── Data Split Quality table (shared with Part 3) ──────────────────────────
    split_tbl = _split_quality_tbl(split) + '\n' + _split_quality_prose(split)

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

        # 3. Variable correlation scatter — must sit on the FIRST page of the
        # report, in the space left under the banner and project table. The box
        # is capped at 46 % of the text height so it fits there whatever the
        # banner's height; LaTeX would otherwise float it to page 2.
        r'\section{Variable Correlation --- Input vs Output}' + '\n'
        + (r'\begin{center}\includegraphics[' + _GFX_MATRIX + r']{'
           + os.path.relpath(paths['data_scatter_vars'], str(out_dir))
           + r'}\end{center}' + '\n'
           if paths.get('data_scatter_vars') else '')

        # 3b. Data statistics right under the matrix — the feature-selection
        # notebook's describe() for inputs and outputs (also in Part 3).
        + ((r'\section{Data Statistics}' + '\n'
            r'Per-variable statistics over all data (train + val + test), as '
            r'\texttt{Train\_set.describe()} reports them in the feature-selection '
            r'notebook: count, mean, standard deviation, minimum, quartiles and maximum.' + '\n\n'
            r'\exhead{Inputs}' + '\n' + _tbl('describe_inputs')
            + r'\exhead{Outputs}' + '\n' + _tbl('describe_outputs'))
           if (_tbl('describe_inputs') or _tbl('describe_outputs')) else '')

        # 4. Model Selection
        + r'\section{Model Selection}' + '\n'
        + _tbl_with_legend(comp_tbl, comp_legend) + '\n\n'

        # 5. Accuracy Assessment
        r'\section{Accuracy Assessment (Q90 \& R\textsuperscript{2})}' + '\n'
        + _tbl_with_legend(metrics_tbl, _LEGEND) + '\n\n'

        # 6. Training History
        + _training_curve_tex() +

        # 7. Predicted vs True — winner model
        rf'\section{{Predicted vs True --- {best_esc}}}' + '\n'
        + winner_scatter_tex + '\n\n'

        # 8. Data Split Quality
        r'\section{Data Split Quality}' + '\n'
        + split_tbl + '\n\n'

        # 9. Overfitting Check
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

def _training_curve(d, plots_dir, artifacts_dir=None):
    """
    Plot the training history of every model that recorded one.

    SF_9 already saves this as an artifact (model_validation/training_curve.py);
    that copy is reused when present so the report and the pipeline cannot
    disagree. Otherwise it is rebuilt here from the saved models, since the
    curves live on the fitted estimator rather than in the metadata.

    Returns the path to use, or None when no model exposes a curve.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import joblib

    if artifacts_dir:
        existing = Path(artifacts_dir) / 'training_curve.png'
        if existing.exists():
            print(f'  training curve: reusing SF_9 artifact {existing.name}')
            return str(existing)

    curves = {}
    for label, path in (d.get('model_files') or {}).items():
        if not path or not Path(path).exists():
            continue
        try:
            model = joblib.load(path)
        except Exception as e:
            print(f'  training curve: could not load {label} ({type(e).__name__})')
            continue

        loss = getattr(model, 'loss_curve_', None)
        if loss is not None and len(loss):
            curves[label] = {
                'loss': list(loss),
                'val': list(getattr(model, 'validation_scores_', None) or []),
                'kind': 'loss',
            }
            continue

        # MultiOutputRegressor(GradientBoosting...): average the per-stage
        # training score across the one-estimator-per-output members.
        inner = getattr(model, 'estimators_', None) or []
        stage = [getattr(e, 'train_score_', None) for e in inner]
        stage = [s for s in stage if s is not None and len(s)]
        if stage:
            n = min(len(s) for s in stage)
            curves[label] = {
                'loss': [float(np.mean([s[i] for s in stage])) for i in range(n)],
                'val': [],
                'kind': 'deviance',
            }

    if not curves:
        print('  training curve: no model exposes one — skipped')
        return None

    has_val = any(c['val'] for c in curves.values())
    ncols = 2 if has_val else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 3.4), squeeze=False)
    ax = axes[0][0]

    for label, c in curves.items():
        ax.plot(range(1, len(c['loss']) + 1), c['loss'], lw=1.2, label=label)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Training loss')
    ax.set_yscale('log')
    ax.grid(alpha=0.3, which='both')
    ax.set_title('Training loss per iteration', fontsize=10)
    ax.legend(fontsize=8)

    if has_val:
        ax2 = axes[0][1]
        vals = []
        for label, c in curves.items():
            if c['val']:
                ax2.plot(range(1, len(c['val']) + 1), c['val'], lw=1.2, label=label)
                vals += c['val']
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Validation score (R²)')
        ax2.grid(alpha=0.3)
        ax2.set_title('Validation score per iteration', fontsize=10)
        ax2.legend(fontsize=8)

        # The first few iterations can sit at R² = -200, which flattens the
        # whole converged region into a line at the top. Clip to the part worth
        # reading and say that is what happened.
        hi = max(vals)
        if min(vals) < -0.5 < hi:
            ax2.set_ylim(-0.05, min(1.02, hi + 0.02))
            ax2.text(0.98, 0.04, f'axis clipped — early iterations reach '
                                 f'{min(vals):.0f}',
                     transform=ax2.transAxes, ha='right', va='bottom',
                     fontsize=7, color='gray')

    plt.tight_layout()
    out = Path(plots_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'training_curve.png'
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)

    summary = ', '.join(f"{k} ({len(v['loss'])} it)" for k, v in curves.items())
    print(f'  plot: training_curve.png  [{summary}]')
    return str(path)


MAX_SCATTER_VARS = 10  # a square grid past this is unreadable on a page


def _styler_df(styler):
    """The DataFrame under a pandas Styler (or the object itself if plain)."""
    import pandas as pd
    return styler.data if hasattr(styler, 'data') else pd.DataFrame(styler)


def _cell_to_tex(v):
    """One table cell to LaTeX, converting prediction_stats' <sup>/<sub> CI markup."""
    import re
    s = str(v)
    m = re.search(r'(.*?)<sup>(.*?)</sup>\s*<sub>(.*?)</sub>(.*)', s)
    if m:
        head, hi, lo, tail = (t.strip() for t in m.groups())
        return rf'{_esc(head)}$^{{{_esc2m(hi)}}}_{{{_esc2m(lo)}}}$'
    s = re.sub(r'<[^>]+>', '', s)          # any other stray markup
    return _esc(s)


def _esc2m(s):
    """Escape for use inside math mode (percent signs mainly)."""
    return s.replace('%', r'\%').replace('&', r'\&')


def _df_to_tex(df, caption=None, font=r'\footnotesize', col_fmt=None,
               cell_colors=None):
    """
    A DataFrame as a LaTeX table. cell_colors: optional callable
    (row_label, col_label, value) -> LaTeX color name or None.
    """
    cols = list(df.columns)
    fmt = col_fmt or ('l' + 'r' * len(cols))
    # A wide table (prediction_stats emits ~15 columns) runs off the page at
    # any fixed font size, so scale it to the line width instead.
    wide = len(cols) > 8
    lines = [r'\begin{center}', font,
             r'\renewcommand{\arraystretch}{1.15}']
    if wide:
        lines.append(r'\resizebox{\linewidth}{!}{%')
    lines += [rf'\begin{{tabular}}{{|{"|".join(fmt)}|}}', r'\hline',
              ' & '.join([r'\textbf{}'] + [rf'\textbf{{{_esc(str(c))}}}' for c in cols])
              + r' \\ \hline']
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            tex = _cell_to_tex(row[c])
            if cell_colors:
                color = cell_colors(idx, c, row[c])
                if color:
                    tex = rf'\cellcolor{{{color}}}{tex}'
            cells.append(tex)
        lines.append(rf'\textbf{{{_esc(str(idx))}}} & ' + ' & '.join(cells) + r' \\ \hline')
    lines += [r'\end{tabular}']
    if wide:
        lines.append('}')
    if caption:
        lines += [r'\par\vspace{2pt}', rf'{{\scriptsize {caption}}}']
    lines += [r'\end{center}']
    return '\n'.join(lines) + '\n'


def _pval_color(v, alpha=0.05):
    """validationlib's own convention: red below alpha, green otherwise."""
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return 'failred' if x < alpha else 'passgreen'


def _table_pair(styler, tex_caption=None, alpha=None):
    """A Styler (or DataFrame) rendered both ways: {'tex': ..., 'html': ...}."""
    df = _styler_df(styler)
    colors = (lambda i, c, v: _pval_color(v, alpha)) if alpha is not None else None
    tex = _df_to_tex(df, caption=tex_caption, cell_colors=colors)
    try:
        html = styler.to_html() if hasattr(styler, 'to_html') else df.to_html()
    except Exception:
        html = df.to_html()
    return {'tex': tex, 'html': html}


def _describe_pair(df, title):
    """
    The feature-selection notebook's Train_set.describe(), transposed so many
    variables fit an A4 page: one row per variable, the standard eight columns.
    """
    import pandas as pd
    desc = df.describe().T
    desc = desc.rename(columns={'count': 'count', '25%': 'P25', '50%': 'P50', '75%': 'P75'})
    shown = desc.copy()
    shown['count'] = shown['count'].map(lambda v: f'{int(v):,}')
    for c in [c for c in shown.columns if c != 'count']:
        shown[c] = shown[c].map(lambda v: f'{v:.4g}')
    tex = _df_to_tex(shown, caption=_esc(title))
    html = (f'<h4>{title}</h4>'
            + shown.to_html(classes='describe-tbl', border=0))
    return {'tex': tex, 'html': html}


def _analysis_plots(best, csv_dir, plots_dir, q90_target, scatter_cfg=None):
    """
    Generate the deep-analysis figures and tables from the validation CSVs.

    Every figure comes from validationlib, called with the same arguments the
    extended validation notebook (validation_template.ipynb) uses, so the
    executive summary, the deep-analysis HTML and validation_output.html all
    share one look. Tables (pandas Stylers from validationlib) are returned in
    stats['tables'] as native LaTeX and HTML rather than screenshots.

    Returns (paths_dict, stats_dict).
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import scipy.stats as st
    from scipy import stats as sc

    import validationlib
    import validationlib.plots
    import validationlib.tables.summary
    import validationlib.tests.dist
    import validationlib.tests.bias
    from validationlib.misc.metrics import DistanceMetrics
    from validationlib.misc.subsampling import bin_data
    from validationlib.plots.nDimensional import scatterplotMatrix

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

    x_val = yt_val = yh_val = None
    if (csv_dir / 'x_val.csv').exists():
        x_val  = pd.read_csv(csv_dir / 'x_val.csv')
        yt_val = pd.read_csv(csv_dir / 'yt_val.csv')
        yh_val = pd.read_csv(csv_dir / 'yh_val.csv')

    outputs = yt_test.columns.tolist()
    inputs  = x_test.select_dtypes(include='number').columns.tolist()

    # The template's two error metrics, defined identically (Residue = y - yhat).
    residue_metric = DistanceMetrics('Residue').define_metric(
        lambda y_true, y_pred: y_true - y_pred)
    abserr_metric = DistanceMetrics('Absolute Error').define_metric(
        lambda y_true, y_pred: np.abs(y_true - y_pred))
    residue_df = residue_metric(yt_test, yh_test)
    abserr_df  = abserr_metric(yt_test, yh_test)

    # Relative error is not part of the template, but the pipeline's Q90
    # requirement is defined on it, so it is computed for the requirement table
    # and one library-styled CDF.
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_vals = abserr_df.values / np.where(np.abs(yt_test.values) > 1e-10,
                                               np.abs(yt_test.values), np.nan)
    relerr_df = pd.DataFrame(rel_vals, columns=outputs)
    q90_results = {o: float(np.nanquantile(relerr_df[o].values, 0.90)) for o in outputs}

    paths, tables = {}, {}

    def _save(fig, name):
        p = plots_dir / name
        fig.savefig(p, dpi=110, bbox_inches='tight')
        plt.close(fig)
        paths[name.replace('.png', '')] = str(p)
        print(f'  plot: {name}')

    def _step(name, fn):
        """Run one figure/table step; a failure loses that item, not the report."""
        try:
            fn()
        except Exception as e:
            print(f'  SKIPPED {name}: {type(e).__name__}: {str(e)[:140]}')

    def _lib_scatter(name, x, y, **kw):
        """
        validationlib scatterplot with a fallback for one-sided color data:
        TwoSlopeNorm demands vmin < vcenter < vmax, so coloring by a strictly
        positive metric (absolute error) with the default center 0 raises.
        """
        try:
            fig = validationlib.plots.scatterplot(x, y, **kw)
        except ValueError:
            c = kw.get('c')
            if c is not None:
                kw = dict(kw, cmapCenter=float(np.nanmedian(_styler_df(c).values)))
                fig = validationlib.plots.scatterplot(x, y, **kw)
            else:
                raise
        _save(fig, name)

    # ── 3.1 DATA OVERVIEW ─────────────────────────────────────────────────────
    x_all  = pd.concat([x_train, x_test] + ([x_val] if x_val is not None else []),
                       ignore_index=True)[inputs]
    yt_all = pd.concat([yt_train, yt_test] + ([yt_val] if yt_val is not None else []),
                       ignore_index=True)

    # The describe() tables from the feature-selection notebook, as asked.
    _step('describe_inputs', lambda: tables.update(
        describe_inputs=_describe_pair(x_all, 'Input variables — Train_set.describe()')))
    _step('describe_outputs', lambda: tables.update(
        describe_outputs=_describe_pair(yt_all, 'Output variables — Train_set.describe()')))

    _step('data_input_hist', lambda: _save(
        validationlib.plots.histogram(
            x_all, xlabel='Input Variable', trimStds=3,
            multiPlotsKwargs={'tight_layout': True}, logscale=True),
        'data_input_hist.png'))

    _step('data_input_cdf', lambda: _save(
        validationlib.plots.cumulative(
            x_all, xlabel='Input Variable', bins=1000,
            quantiles=[0.1, 0.50, 0.90, 0.95, 0.99]),
        'data_input_cdf.png'))

    _step('data_output_hist', lambda: _save(
        validationlib.plots.histogram(
            yt_all, xlabel='Output Variable',
            multiPlotsKwargs={'tight_layout': True}, logscale=True),
        'data_output_hist.png'))

    _step('data_output_cdf', lambda: _save(
        validationlib.plots.cumulative(
            yt_all, xlabel='Ground truth', bins=1000,
            quantiles=[0.1, 0.50, 0.90, 0.95, 0.99]),
        'data_output_cdf.png'))

    # Scatterplot matrix over a configurable variable list (square N x N grid,
    # so which variables go in is a choice — SF_9 metadata scatter_variables).
    def _scatter_matrix():
        cfg = scatter_cfg or {}
        chosen = [v for v in (cfg.get('variables') or [])
                  if v in x_all.columns or v in yt_all.columns]
        if not chosen:
            half = MAX_SCATTER_VARS // 2
            n_out_keep = min(len(outputs), max(half, MAX_SCATTER_VARS - len(inputs)))
            n_inp_keep = min(len(inputs), MAX_SCATTER_VARS - n_out_keep)
            chosen = list(inputs[:n_inp_keep]) + list(outputs[:n_out_keep])
            if n_inp_keep < len(inputs) or n_out_keep < len(outputs):
                print(f'  scatter matrix: {len(inputs)} inputs and {len(outputs)} outputs '
                      f'do not fit a square grid — showing {n_inp_keep} and {n_out_keep}. '
                      f'Set Model_Validation.scatter_variables in SF_9 metadata to choose.')
        elif len(chosen) > MAX_SCATTER_VARS:
            print(f'  scatter matrix: {len(chosen)} variables requested, using the '
                  f'first {MAX_SCATTER_VARS}')
            chosen = chosen[:MAX_SCATTER_VARS]

        print(f'  scatter matrix variables: {list(chosen)}')
        combined = pd.concat([x_all, yt_all], axis=1)
        fig = scatterplotMatrix(combined[chosen].to_numpy(dtype=float), list(chosen),
                                method=(cfg.get('method') or 'scatter'),
                                s=1, figsize=min(8.0, 1.5 * len(chosen) + 1.5))
        for ax in fig.get_axes():
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            ax.tick_params(labelsize=6)
        # The library writes each variable name once, centred on its diagonal
        # cell, at the default size; a long name such as
        # bottom_transition_location spills into the neighbours. Break it at
        # the underscores and size it to the cell.
        n = len(chosen)
        axes = fig.get_axes()
        for i, name in enumerate(chosen):
            for t in axes[i * n + i].texts:
                if t.get_text() == name and len(name) > 10:
                    parts, lines, cur = name.split('_'), [], ''
                    for p_ in parts:
                        if cur and len(cur) + 1 + len(p_) > 12:
                            lines.append(cur); cur = p_
                        else:
                            cur = f'{cur}_{p_}' if cur else p_
                    lines.append(cur)
                    t.set_text('\n'.join(lines))
                    t.set_fontsize(7 if n <= 8 else 6)
        fig.suptitle('Variable Correlation', fontsize=11)
        _save(fig, 'data_scatter_vars.png')
    _step('data_scatter_vars', _scatter_matrix)

    # ── 3.2 TRAIN-TEST SPLIT ──────────────────────────────────────────────────
    _step('split_output_dists', lambda: _save(
        validationlib.plots.doubleHistogram(
            yt_train, yt_test, xlabel='Output', x1label='Train', x2label='Test',
            multiPlotsKwargs={'tight_layout': True}, logscale=True),
        'split_output_dists.png'))

    _step('split_input_dists', lambda: _save(
        validationlib.plots.doubleHistogram(
            x_train[inputs], x_test[inputs], xlabel='Input',
            x1label='Train', x2label='Test',
            multiPlotsKwargs={'tight_layout': True}),
        'split_input_dists.png'))

    _step('split_ks_ad', lambda: tables.update(split_ks_ad=_table_pair(
        validationlib.tests.dist.dist_similarity_table(
            yt_train, yt_test, title='p-values', tests=['KS', 'AD']),
        tex_caption='Train vs test p-values. Red: $p<0.05$ (distributions differ).',
        alpha=0.05)))

    # KS results per output, for the summary prose elsewhere in the report.
    ks_results = {}
    for o in outputs:
        ks_stat, ks_p = sc.ks_2samp(yt_train[o].values, yt_test[o].values)
        ks_results[o] = {'stat': float(ks_stat), 'p': float(ks_p)}

    # ── 3.3 ERROR QUANTIFICATION P(E) ─────────────────────────────────────────
    statistics = {'mean': np.mean, 'median': [np.percentile, {'q': 50}],
                  'std': np.std, 'IQR': st.iqr,
                  'kurtosis': st.kurtosis, 'skewness': st.skew}

    _step('res_stats', lambda: tables.update(res_stats=_table_pair(
        validationlib.tables.summary.prediction_stats(
            residue_df, statistics=statistics, precision=3, nsim=100,
            method='Bootstrap'),
        tex_caption='Residue statistics with bootstrap confidence intervals '
                    '(superscript/subscript bounds).')))

    _step('abs_stats', lambda: tables.update(abs_stats=_table_pair(
        validationlib.tables.summary.prediction_stats(
            abserr_df, statistics=statistics, precision=3, nsim=100,
            method='Bootstrap'),
        tex_caption='Absolute-error statistics with bootstrap confidence intervals.')))

    _step('err_residue_hist', lambda: _save(
        validationlib.plots.histogram(
            residue_df, xlabel=residue_metric.name,
            multiPlotsKwargs={'tight_layout': True}),
        'err_residue_hist.png'))

    _step('err_residue_cdf', lambda: _save(
        validationlib.plots.cumulative(
            residue_df, xlabel=residue_metric.name, bins=1000,
            quantiles=[0.1, 0.50, 0.90, 0.95, 0.99]),
        'err_residue_cdf.png'))

    _step('err_abserr_hist', lambda: _save(
        validationlib.plots.histogram(
            abserr_df, xlabel=abserr_metric.name,
            multiPlotsKwargs={'tight_layout': True}, logscale=True),
        'err_abserr_hist.png'))

    _step('err_relerr_cdf', lambda: _save(
        validationlib.plots.cumulative(
            relerr_df.dropna(), xlabel='Relative Absolute Error', bins=1000,
            quantiles=[0.90]),
        'err_relerr_cdf.png'))

    _step('err_true_vs_pred_dist', lambda: _save(
        validationlib.plots.doubleHistogram(
            yt_test, yh_test, xlabel='Output', x1label='True', x2label='Predicted',
            logscale=True),
        'err_true_vs_pred_dist.png'))

    _step('true_pred_pvals', lambda: tables.update(true_pred_pvals=_table_pair(
        validationlib.tests.dist.dist_similarity_table(
            yt_test, yh_test, title='Hypothesis Tests Results (p-value)',
            tests=['AD', 'KS']),
        tex_caption='True vs predicted distribution tests. Red: $p<0.05$.',
        alpha=0.05)))

    # The template's predicted-vs-true scatter, reused as the Part 1 figure.
    _step('winner_pred_true', lambda: _lib_scatter(
        'winner_pred_true.png', yt_test, yh_test,
        c=residue_df, correlationAxis='fit', fit_info='default',
        significant_figures=3, xlabel='True', ylabel='Predicted',
        clabel=residue_metric.name, cmapCenter=0,
        multiPlotsKwargs={'tight_layout': True}))

    _step('true_pred_hist2d', lambda: _save(
        validationlib.plots.hist2D(
            yt_test, yh_test, bins=100, xlabel='True', ylabel='Predicted',
            scale='frequency', correlationAxis='fit', logscale=True,
            figHsize=7, figAspectRatio=2),
        'true_pred_hist2d.png'))

    # ── 3.4 P(E|X) ────────────────────────────────────────────────────────────
    for o in outputs:
        safe = o.replace(' ', '_').replace('/', '_')
        _step(f'pex_violin_res_{safe}', lambda o=o, safe=safe: _save(
            validationlib.plots.violinPlot(
                x_test[inputs], residue_df[[o]], bins='sturges', var_y=o,
                xlabel='Bins', ylabel=residue_metric.name, showextrema=True,
                trimStds=3, multiPlotsKwargs={'tight_layout': True}),
            f'pex_violin_res_{safe}.png'))
        _step(f'pex_violin_abs_{safe}', lambda o=o, safe=safe: _save(
            validationlib.plots.violinPlot(
                x_test[inputs], abserr_df[[o]], bins='sturges', var_y=o,
                xlabel='Bins', ylabel=abserr_metric.name, showextrema=True,
                trimStds=3, multiPlotsKwargs={'tight_layout': True}),
            f'pex_violin_abs_{safe}.png'))

    # Kruskal-Wallis bias table on Sturges-binned inputs, exactly as the
    # template's bias_detection_table cell.
    kw_pvals = None
    def _bias_table():
        nonlocal kw_pvals
        df_binned = pd.DataFrame(index=x_test.index)
        binned_vars = []
        for in_var in inputs:
            _, df_binned[in_var + '_binned'] = bin_data(
                x_test[in_var].values, bins='sturges')
            binned_vars.append(in_var + '_binned')
        df_binned = df_binned.join(residue_df)
        styler = validationlib.tests.bias.bias_detection_table(
            df_binned, binned_vars, list(residue_df.columns),
            method='kruskal', info=False)
        tables['bias_kruskal'] = _table_pair(
            styler,
            tex_caption='Kruskal-Wallis p-values on Sturges-binned inputs. '
                        'Red: $p<0.05$ (error depends on that input).',
            alpha=0.05)
        raw = _styler_df(styler).apply(pd.to_numeric, errors='coerce')
        # Orient as (inputs, outputs) whichever way the library returns it.
        if list(raw.index) == binned_vars or len(raw.index) == len(binned_vars):
            kw_pvals = raw.values
        else:
            kw_pvals = raw.values.T
    _step('bias_kruskal', _bias_table)

    # ── 3.5 P(E|Y) ────────────────────────────────────────────────────────────
    _step('pey_residue_scatter', lambda: _lib_scatter(
        'pey_residue_scatter.png', yt_test, residue_df,
        c=residue_df, clabel=residue_metric.name, correlationAxis='fit',
        xlabel='Output', ylabel=residue_metric.name, trimStds=3, cmapCenter=0,
        multiPlotsKwargs={'tight_layout': True}))

    _step('pey_residue_violin', lambda: _save(
        validationlib.plots.violinPlot(
            yt_test, residue_df, bins='sturges', xlabel='Bins',
            ylabel=residue_metric.name, showextrema=True, trimStds=2,
            multiPlotsKwargs={'tight_layout': True}),
        'pey_residue_violin.png'))

    _step('pey_abserr_scatter', lambda: _lib_scatter(
        'pey_abserr_scatter.png', yt_test, abserr_df,
        c=abserr_df, clabel=abserr_metric.name, correlationAxis='fit',
        xlabel='Output', ylabel=abserr_metric.name, trimStds=3, cmapCenter=0,
        multiPlotsKwargs={'tight_layout': True}))

    _step('pey_abserr_violin', lambda: _save(
        validationlib.plots.violinPlot(
            yt_test, abserr_df, bins='sturges', xlabel='Bins',
            ylabel=abserr_metric.name, showextrema=True, trimStds=2,
            multiPlotsKwargs={'tight_layout': True}),
        'pey_abserr_violin.png'))

    # ── 3.6 UNCERTAINTY ───────────────────────────────────────────────────────
    # The template's global binned uncertainty model (bins=1), trained on the
    # validation split and covered on test. Needs a validation split.
    def _uncertainty():
        from scipy.sparse import csr_matrix
        from validationlib.tests.interval import BinnedUncertaintyModel, ModelCoverage

        if yt_val is None or len(yt_val) < 20:
            print('  uncertainty: no validation split in the CSVs — skipped')
            return

        yt_valtest = pd.concat([yt_val, yt_test], axis=0, ignore_index=True)
        yh_valtest = pd.concat([yh_val, yh_test], axis=0, ignore_index=True)
        abserr_valtest = abserr_metric(yt_valtest, yh_valtest)

        val_mask = np.zeros(yt_valtest.shape, dtype=bool)
        val_mask[:len(yt_val)] = True
        test_mask = ~val_mask
        val_mask, test_mask = csr_matrix(val_mask), csr_matrix(test_mask)

        n_cal = yt_val.shape[0]
        min_elems = max(10, n_cal // 10)

        model = BinnedUncertaintyModel(
            percentile_range=95, model_type='eqspaced_bins',
            ci_type='right_tailed',
            method_kwargs={'bins': 1, 'nsim': 100, 'min_elems': min_elems,
                           'conf': 0.95},
            matched_analysis=True)
        model.train(yh_valtest, abserr_valtest, val_mask,
                    x_variables=yt_test.columns, y_variables=yt_test.columns)

        cov_model = ModelCoverage(matched_analysis=True, conf=0.95)
        cov_model.compute_coverage(yh_valtest, abserr_valtest, test_mask, model,
                                   x_variables=yt_test.columns,
                                   y_variables=yt_test.columns)

        # coverage_plot returns a LIST of figures (one per conditioned variable).
        figs = model.coverage_plot(
            yh_valtest, abserr_valtest, val_mask, test_mask, cov_model,
            y_label=abserr_metric.name, plot_type='histogram',
            hist_data='test', bins=500, logscale=True)
        if not isinstance(figs, (list, tuple)):
            figs = [figs]
        for k, fig in enumerate(f for f in figs if f is not None):
            _save(fig, 'uncertainty_coverage.png' if k == 0
                       else f'uncertainty_coverage_{k + 1}.png')

        cov_tables = cov_model.get_tables()
        if cov_tables:
            merged = pd.concat(cov_tables, axis=0) if len(cov_tables) > 1 else cov_tables[0]
            tables['uncertainty_coverage_tbl'] = _table_pair(
                merged, tex_caption='Empirical coverage of the 95th-percentile '
                                    'uncertainty bound on the test split.')
    _step('uncertainty', _uncertainty)

    stats = dict(
        q90_results=q90_results,
        res_stats={o: {'mean': float(residue_df[o].mean()),
                       'std': float(residue_df[o].std()),
                       'p10': float(np.percentile(residue_df[o], 10)),
                       'p90': float(np.percentile(residue_df[o], 90))}
                   for o in outputs},
        kw_pvals=kw_pvals,
        inputs=inputs,
        outputs=outputs,
        ks_results=ks_results,
        cov_rows=[],
        tables=tables,
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

    def _tbl(key):
        """A native LaTeX table produced alongside the validationlib figures."""
        return (stats.get('tables') or {}).get(key, {}).get('tex', '')

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
        r'Statistical summary of inputs and outputs over all data (train + val + test), '
        r'as reported by \texttt{describe()} in the feature-selection notebook: '
        r'count, mean, standard deviation, minimum, quartiles and maximum per variable. '
        r'All figures in this part are drawn by \texttt{validationlib}, matching the '
        r'extended validation notebook.' + '\n\n'
        r'\exhead{Input Statistics}' + '\n'
        + _tbl('describe_inputs') +
        r'\exhead{Input Histograms}' + '\n'
        + _fig('data_input_hist') +
        r'\exhead{Input Cumulative Distributions}' + '\n'
        + _fig('data_input_cdf') +
        r'\exhead{Output Statistics}' + '\n'
        + _tbl('describe_outputs') +
        r'\exhead{Output Histograms}' + '\n'
        + _fig('data_output_hist') +
        r'\exhead{Output Cumulative Distributions}' + '\n'
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
        + _tbl('split_ks_ad')
        + r'\exhead{Split Quality --- Voxel Tesselation Proximity (p-hacking check)}' + '\n'
        r'Beyond matching distributions, a split can still flatter the model if test '
        r'points sit right next to training points (\textbf{p-hacking}) or probe regions '
        r'the model never saw (\textbf{isolated}, \textbf{residual voxel}). The VTPM method '
        r'from \texttt{validationlib} (SF\_9 cell 9.0) classifies every test point.' + '\n\n'
        + _split_quality_tbl(d.get('split')) + '\n\n'
        + _split_quality_prose(d.get('split')) + '\n\n'

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
        r'Bootstrap confidence bounds shown as superscript (upper) and subscript (lower).' + '\n'
        + _tbl('res_stats') +
        r'\exhead{Absolute Error Statistics Table}' + '\n'
        + _tbl('abs_stats') +
        r'\exhead{Residue Histograms}' + '\n'
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
        r'AD and KS tests comparing the distribution of true vs.\ predicted values on the test set. '
        r'A significant difference indicates the model is not reproducing the output distribution faithfully.' + '\n'
        + _fig('err_true_vs_pred_dist')
        + _tbl('true_pred_pvals') +
        r'\exhead{True vs Predicted --- 2D Histogram}' + '\n'
        r'Log-scaled frequency with a linear fit; the normalized slope should be close to 1.' + '\n'
        + _fig('true_pred_hist2d')

        # 3.4 P(E|X)
        + r'\clearpage' + '\n'
        r'\section{P(E|X) --- Error Conditional on Inputs}' + '\n'
        r'\label{sec:d4}' + '\n\n'
        r'We study whether the model error depends on the input variables. '
        r'Two approaches are used: (1) a Kruskal-Wallis (KW) test on binned inputs '
        r'(Sturges rule for bin count), and (2) violin plots showing the error distribution '
        r'within each input bin.' + '\n\n'
        + bias_prose + '\n\n'
        r'\exhead{Bias Detection Table (KW p-values)}' + '\n'
        r'Green ($p\geq 0.05$) = residue uniform across input bins; '
        r'red ($p<0.05$) = significant bias.' + '\n'
        + _tbl('bias_kruskal')
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
        r'A global binned uncertainty model (95th percentile, right-tailed) is trained on the '
        r'validation split of the absolute error and evaluated on the held-out test split, '
        r'as in the extended validation notebook. Coverage is the proportion of test points '
        r'whose error falls inside the predicted bound; it should be close to 95\%.' + '\n\n'
        r'\exhead{Coverage Plot}' + '\n'
        + _fig('uncertainty_coverage') +
        r'\exhead{Coverage Table}' + '\n'
        + _tbl('uncertainty_coverage_tbl')
    )


# ── HTML export ────────────────────────────────────────────────────────────────

def _generate_html(d, best, paths, stats, out_dir):
    uc       = d['use_case']
    date_str = datetime.today().strftime('%d %B %Y')

    def _img(name, max_px=900):
        p = paths.get(name, '')
        if not p or not Path(p).exists(): return ''
        data = base64.b64encode(Path(p).read_bytes()).decode()
        return (f'<img src="data:image/png;base64,{data}" '
                f'style="width:90%;max-width:{max_px}px;display:block;margin:12px auto;">')

    q90_res  = stats.get('q90_results', {})
    q90_target = d['q90_target']
    outputs    = d['outputs']

    def _tblh(key, heading=None):
        """A native HTML table produced alongside the validationlib figures."""
        html = (stats.get('tables') or {}).get(key, {}).get('html', '')
        if not html:
            return ''
        return (f'<h3>{heading}</h3>{html}' if heading else html)

    def _q90_tbl():
        rows = ''
        for o in outputs:
            v  = q90_res.get(o)
            vs = f'{v:.4f}' if v is not None else '—'
            ok = v is not None and v < q90_target
            c  = '#c6efce' if ok else '#ffc7ce'
            rows += f'<tr><td>{o}</td><td style="background:{c}">{vs}</td><td>{q90_target:.0%}</td><td style="background:{c}"><b>{"PASS" if ok else "FAIL"}</b></td></tr>'
        return f'<table><tr><th>Output</th><th>Q90</th><th>Target</th><th>Status</th></tr>{rows}</table>'

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

    def _full_reports_nav():
        """
        Link the per-model reports that validation_script.py produces from
        validation_template.ipynb. Those carry the full analysis — every model,
        plus bias quantification, convex-hull/VTP and uncertainty modelling that
        this summary does not reproduce. They live in the same folder, so
        relative links keep working if the folder is moved or zipped.
        """
        found = sorted(Path(out_dir).glob('*_validation_output.html'))
        if not found:
            return (
                '<h2 id="dfull">Full per-model validation reports</h2>'
                '<p class="meta">None found in this folder. They are produced by '
                'cell 9.5 of SF_9 (<code>validation_script.py</code> running '
                '<code>validation_template.ipynb</code>) as '
                '<code>&lt;model&gt;_validation_output.html</code>.</p>'
            )
        items = ''.join(
            f'<li><a href="{f.name}">{f.name.replace("_validation_output.html", "")}</a>'
            f' &nbsp;<span class="meta">({f.stat().st_size / 1e6:.1f} MB)</span></li>'
            for f in found
        )
        return (
            '<h2 id="dfull">Full per-model validation reports</h2>'
            '<p class="meta">Complete output of <code>validation_template.ipynb</code>, '
            'one per trained model — deeper than the summary below.</p>'
            f'<ul>{items}</ul>'
        )

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>Deep Analysis — {uc}</title>
<style>{css}</style></head><body>
<h1>Deep Analysis &mdash; {best}</h1>
<p class="meta">Use case: <b>{uc}</b> &nbsp;|&nbsp; {date_str} &nbsp;|&nbsp; Mirrors validation_template.ipynb</p>

<nav><b>Contents</b><ul>
<li><a href="#dtrain">3.0 Training History</a></li>
<li><a href="#d0">3.0b Variable Correlation</a></li>
<li><a href="#d1">3.1 Data Overview</a></li>
<li><a href="#d2">3.2 Train-Test Split Analysis</a></li>
<li><a href="#d3">3.3 Error Quantification &mdash; P(E)</a></li>
<li><a href="#d4">3.4 P(E|X) &mdash; Error Conditional on Inputs</a></li>
<li><a href="#d5">3.5 P(E|Y) &mdash; Error Conditional on Outputs</a></li>
<li><a href="#d6">3.6 Uncertainty Analysis</a></li>
<li><a href="#dfull">Full per-model validation reports</a></li>
</ul></nav>

{_full_reports_nav()}

{_section("dtrain","3.0 Training History",
    _img("training_curve", max_px=620) +
    "<p class='meta'>Training loss per iteration, log scale, with a validation "
    "panel for models trained with early stopping.</p>"
) if paths.get("training_curve") else ""}

{_section("d0","3.0b Variable Correlation &mdash; Input vs Output",
    _img("data_scatter_vars")
)}

{_section("d1","3.1 Data Overview",
    _tblh("describe_inputs", "Input Statistics — describe()") +
    "<h3>Input Histograms</h3>" + _img("data_input_hist") +
    "<h3>Input CDFs</h3>" + _img("data_input_cdf") +
    _tblh("describe_outputs", "Output Statistics — describe()") +
    "<h3>Output Histograms</h3>" + _img("data_output_hist") +
    "<h3>Output CDFs</h3>" + _img("data_output_cdf")
)}

{_section("d2","3.2 Train-Test Split Analysis",
    "<h3>Output Distributions</h3>" + _img("split_output_dists") +
    "<h3>Input Distributions</h3>" + _img("split_input_dists") +
    _tblh("split_ks_ad", "KS &amp; AD Test Table")
)}

{_section("d3","3.3 Error Quantification &mdash; P(E)",
    _tblh("res_stats", "Residue Statistics") +
    _tblh("abs_stats", "Absolute Error Statistics") +
    "<h3>Residue Histograms</h3>" + _img("err_residue_hist") +
    "<h3>Residue CDFs</h3>" + _img("err_residue_cdf") +
    "<h3>Absolute Error Histograms</h3>" + _img("err_abserr_hist") +
    "<h3>Relative Error CDFs — Q90 Requirement</h3>" + _img("err_relerr_cdf") + _q90_tbl() +
    "<h3>True vs Predicted Distributions</h3>" + _img("err_true_vs_pred_dist") +
    _tblh("true_pred_pvals") +
    "<h3>True vs Predicted — 2D Histogram</h3>" + _img("true_pred_hist2d") +
    "<h3>Predicted vs True (scatter, colored by residue)</h3>" + _img("winner_pred_true")
)}

{_section("d4","3.4 P(E|X) &mdash; Error Conditional on Inputs",
    _tblh("bias_kruskal", "Bias Detection Table (KW p-values)") +
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
    "<h3>Coverage Plot</h3>" + _img("uncertainty_coverage") +
    _tblh("uncertainty_coverage_tbl", "Coverage Table")
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
        + _part1(d, best, rows, scatter_paths, paths, out_dir, stats)
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
    paths, stats = _analysis_plots(best, csv_dir, plots_dir, d['q90_target'],
                                   d.get('scatter_cfg'))

    curve = _training_curve(d, plots_dir, artifacts_dir)
    if curve:
        paths['training_curve'] = curve

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
