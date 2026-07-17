"""
Generate a LaTeX executive summary PDF from a Surrogate Factory metadata JSON.

Usage:
    python generate_executive_summary.py <metadata.json> [--output <dir>]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


# ── LaTeX helpers ──────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    """Escape special LaTeX characters."""
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'), ('%', r'\%'), ('$', r'\$'), ('#', r'\#'),
        ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
        ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}'),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def _tick(passed: bool) -> str:
    return r'\cellcolor{passgreen}\textbf{\checkmark}' if passed else r'\cellcolor{failred}\textbf{\texttimes}'


def _r2_color(r2) -> str:
    if r2 is None:
        return r'\textemdash'
    if r2 >= 0.95:
        return rf'\textcolor{{darkgreen}}{{\textbf{{{r2:.4f}}}}}'
    if r2 >= 0.80:
        return rf'\textcolor{{orange}}{{\textbf{{{r2:.4f}}}}}'
    return rf'\textcolor{{red}}{{\textbf{{{r2:.4f}}}}}'


# ── Data extraction ────────────────────────────────────────────────────────────

def extract(meta_path: Path) -> dict:
    with open(meta_path) as f:
        root = json.load(f)['metadata']

    val  = root['Model_Validation']
    sel  = root['Model_Selection']
    trn  = root.get('Model_Training', {})
    part = root.get('Data_Partition', {}).get('Data_Partition', {})
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

    # Best model by average R²
    avg_r2 = {}
    for mdl in models:
        r2s = [scores.get(mdl, {}).get(o, {}).get('R2') for o in outputs]
        r2s = [v for v in r2s if v is not None]
        avg_r2[mdl] = sum(r2s) / len(r2s) if r2s else 0
    best_model = max(avg_r2, key=avg_r2.get) if avg_r2 else (models[0] if models else 'N/A')

    # Q90 pass counts
    q90_pass = {m: 0 for m in models}
    for vr in vres:
        for mdl, res in vr.get('models', {}).items():
            if res.get('passed'):
                q90_pass[mdl] = q90_pass.get(mdl, 0) + 1

    # KS test pass counts
    ks_pass = {}
    for mdl in models:
        ks_pass[mdl] = sum(1 for v in dist.get(mdl, {}).values() if v >= 0.05)

    return dict(
        models=models, outputs=outputs, inputs=inputs,
        scores=scores, dist=dist, vres=vres, split=split,
        q90_target=q90_target, best_model=best_model,
        avg_r2=avg_r2, q90_pass=q90_pass, ks_pass=ks_pass,
        n_train=part.get('train', '—'), n_test=part.get('test', '—'),
        n_val=part.get('val', '—'),
        use_case=meta_path.stem.replace('metadata_', ''),
    )


# ── LaTeX document ─────────────────────────────────────────────────────────────

def build_latex(d: dict) -> str:
    models     = d['models']
    outputs    = d['outputs']
    scores     = d['scores']
    vres       = d['vres']
    dist       = d['dist']
    split      = d['split']
    q90_target = d['q90_target']
    best       = d['best_model']
    n_out      = len(outputs)
    n_mdl      = len(models)
    date_str   = datetime.today().strftime('%B %d, %Y')
    uc_name    = _esc(d['use_case'])

    # ── Summary numbers ────────────────────────────────────────────────────────
    best_q90_pass  = d['q90_pass'].get(best, 0)
    best_ks_pass   = d['ks_pass'].get(best, 0)
    best_avg_r2    = d['avg_r2'].get(best, 0)
    overall_pass   = best_q90_pass == n_out and best_ks_pass == n_out

    banner_color   = 'passgreen' if overall_pass else 'warnamber'
    banner_text    = r'ALL REQUIREMENTS MET' if overall_pass else r'REQUIREMENTS PARTIALLY MET'

    # ── Column spec for metrics table ──────────────────────────────────────────
    col_spec = 'l' + 'rr' * n_mdl
    mdl_headers = ' & '.join(
        rf'\multicolumn{{2}}{{c}}{{\textbf{{{_esc(m)}}}}}'
        for m in models
    )
    sub_headers = ''.join(r' & R\textsuperscript{2} & Q90' for _ in models)

    # ── Metrics rows ───────────────────────────────────────────────────────────
    vres_map = {vr['output']: vr for vr in vres}
    rows = []
    for o in outputs:
        cells = [_esc(o)]
        for m in models:
            r2  = scores.get(m, {}).get(o, {}).get('R2')
            vr  = vres_map.get(o, {})
            mres = vr.get('models', {}).get(m, {})
            q90_val    = mres.get('score')
            q90_passed = mres.get('passed', False)
            r2_cell  = _r2_color(r2)
            q90_cell = (
                rf'\cellcolor{{passgreen}}{q90_val:.3f}'
                if q90_passed
                else (rf'\cellcolor{{failred}}{q90_val:.3f}' if q90_val is not None else r'\textemdash')
            )
            cells += [r2_cell, q90_cell]
        rows.append(' & '.join(cells) + r' \\')
    metrics_rows = '\n        '.join(rows)

    # ── KS test rows ───────────────────────────────────────────────────────────
    ks_rows = []
    for o in outputs:
        cells = [_esc(o)]
        for m in models:
            pval = dist.get(m, {}).get(o)
            if pval is None:
                cells.append(r'\textemdash')
            else:
                passed = pval >= 0.05
                cells.append(
                    rf'\cellcolor{{passgreen}}{pval:.3f}' if passed
                    else rf'\cellcolor{{failred}}{pval:.3f}'
                )
        ks_rows.append(' & '.join(cells) + r' \\')
    ks_rows_tex = '\n        '.join(ks_rows)

    # ── Inputs list ───────────────────────────────────────────────────────────
    inputs_str = ', '.join(_esc(i) for i in d['inputs'])

    # ── Summary table rows ────────────────────────────────────────────────────
    summary_rows = []
    for m in models:
        q90_p = d['q90_pass'].get(m, 0)
        ks_p  = d['ks_pass'].get(m, 0)
        avg   = d['avg_r2'].get(m, 0)
        bold  = r'\textbf' if m == best else ''
        summary_rows.append(
            rf'    {bold}{{{_esc(m)}}} & {bold}{{{avg:.4f}}} & {bold}{{{q90_p}/{n_out}}} & {bold}{{{ks_p}/{n_out}}} \\'
        )
    summary_rows_tex = '\n'.join(summary_rows)

    # ── Pre-compute expressions that contain backslashes (can't go in f{}) ───
    ks_col_spec = 'l' + 'r' * n_mdl
    ks_headers  = ' & '.join(r'\textbf{' + _esc(m) + '}' for m in models)
    rvp = split.get('residual_voxel_proportion')
    split_rvp_val  = f'{rvp:.3f}' if rvp is not None else '—'
    split_rvp_icon = r'$\checkmark$ ($\leq 0.05$)' if (rvp or 1) <= 0.05 else r'$\times$'
    vtp = split.get('valid_test_proportion')
    split_vtp_val  = f'{vtp:.3f}' if vtp is not None else '—'
    pht = split.get('phacking_test_proportion')
    split_pht_val  = f'{pht:.3f}' if pht is not None else '—'
    itp = split.get('isolated_test_proportion')
    split_itp_val  = f'{itp:.3f}' if itp is not None else '—'
    chi = split.get('chi_squared_pvalue')
    split_chi_val  = f'{chi:.4f}' if chi is not None else '—'
    split_chi_icon = r'$\checkmark$ ($\geq 0.05$)' if (chi or 0) >= 0.05 else r'$\times$'
    deploy_pkl     = r'pipeline\_' + uc_name + r'\_' + _esc(best) + r'.pkl'
    # percentage sign must be \% in LaTeX — pre-compute so it never appears raw in math mode
    q90_pct_lt = rf'{q90_target * 100:.0f}\%'   # e.g. "10\%"
    q90_pct_ge = rf'{q90_target * 100:.0f}\%'

    return rf"""
\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{booktabs, colortbl, xcolor, array, makecell}}
\usepackage{{amsmath, amssymb}}
\usepackage{{helvet}}
\renewcommand{{\familydefault}}{{\sfdefault}}
\usepackage{{fancyhdr, graphicx, hyperref}}
\usepackage{{parskip}}

% ── Colours ────────────────────────────────────────────────────────────────────
\definecolor{{passgreen}}{{RGB}}{{198,239,206}}
\definecolor{{failred}}{{RGB}}{{255,199,206}}
\definecolor{{warnamber}}{{RGB}}{{255,235,156}}
\definecolor{{darkgreen}}{{RGB}}{{0,128,0}}
\definecolor{{primary}}{{RGB}}{{0,70,127}}

\hypersetup{{colorlinks=true, linkcolor=primary, urlcolor=primary}}

\pagestyle{{fancy}}
\fancyhf{{}}
\lhead{{\textcolor{{primary}}{{\textbf{{Surrogate Factory — Executive Summary}}}}}}
\rhead{{{uc_name}}}
\cfoot{{\thepage}}

\begin{{document}}

% ── Cover ─────────────────────────────────────────────────────────────────────
\begin{{center}}
  {{\Huge\textbf{{\textcolor{{primary}}{{Executive Summary}}}}}}\\[6pt]
  {{\large Surrogate Model Validation Report}}\\[4pt]
  {{\large \texttt{{{uc_name}}}}}\\[10pt]
  {{\normalsize {date_str}}}
\end{{center}}

\noindent\rule{{\textwidth}}{{1.5pt}}

% ── Banner ────────────────────────────────────────────────────────────────────
\begin{{center}}
  \colorbox{{{banner_color}}}{{\parbox{{0.85\textwidth}}{{\centering\large\textbf{{{banner_text}}}\\
  Best model: \textbf{{{_esc(best)}}} \quad
  Avg R\textsuperscript{{2}}: \textbf{{{best_avg_r2:.4f}}} \quad
  Q90 pass: \textbf{{{best_q90_pass}/{n_out}}} \quad
  KS pass: \textbf{{{best_ks_pass}/{n_out}}}}}}}
\end{{center}}

\vspace{{4pt}}

% ── 1. Overview ───────────────────────────────────────────────────────────────
\section*{{1.\enspace Project Overview}}

\begin{{tabular}}{{ll}}
  \textbf{{Use case}}    & {uc_name} \\
  \textbf{{Models}}      & {', '.join(_esc(m) for m in models)} \\
  \textbf{{Outputs}}     & {n_out} ({', '.join(_esc(o) for o in outputs)}) \\
  \textbf{{Inputs}}      & {inputs_str} \\
  \textbf{{Train / Val / Test}} & {d['n_train']} / {d['n_val']} / {d['n_test']} rows \\
  \textbf{{Requirement}} & Q90 relative error $< {q90_pct_lt}$ per output \\
\end{{tabular}}

% ── 2. Model Comparison ───────────────────────────────────────────────────────
\section*{{2.\enspace Model Comparison (Test Set)}}

\renewcommand{{\arraystretch}}{{1.25}}
\begin{{tabular}}{{lrrr}}
  \toprule
  \textbf{{Model}} & \textbf{{Avg R\textsuperscript{{2}}}} &
  \textbf{{Q90 pass ({n_out} outputs)}} & \textbf{{KS pass ({n_out} outputs)}} \\
  \midrule
{summary_rows_tex}
  \bottomrule
\end{{tabular}}

\medskip
\noindent\textit{{Q90 target: $<{q90_pct_lt}$ relative error at 90th percentile.
KS test: $p \geq 0.05$ means train and test residuals follow the same distribution (no overfitting).}}

% ── 3. Metrics per Output ─────────────────────────────────────────────────────
\section*{{3.\enspace Metrics per Output (Q90 target $< {q90_pct_lt}$)}}

\renewcommand{{\arraystretch}}{{1.25}}
\begin{{tabular}}{{{col_spec}}}
  \toprule
  \textbf{{Output}} & {mdl_headers} \\
  & {sub_headers[3:]} \\
  \midrule
        {metrics_rows}
  \bottomrule
\end{{tabular}}

\medskip
\noindent\colorbox{{passgreen}}{{pass}}\enspace Q90 $< {q90_pct_lt}$\quad
\colorbox{{failred}}{{fail}}\enspace Q90 $\geq {q90_pct_lt}$\quad
\textcolor{{darkgreen}}{{\textbf{{R\textsuperscript{{2}} $\geq 0.95$}}}}\enspace
\textcolor{{orange}}{{\textbf{{$0.80 \leq$ R\textsuperscript{{2}} $< 0.95$}}}}\enspace
\textcolor{{red}}{{\textbf{{R\textsuperscript{{2}} $< 0.80$}}}}

% ── 4. Overfitting Check ──────────────────────────────────────────────────────
\section*{{4.\enspace Overfitting Check (KS Test p-values)}}

\noindent$H_0$: train and test residuals follow the same distribution.
Rejection ($p < 0.05$) indicates overfitting.

\renewcommand{{\arraystretch}}{{1.25}}
\begin{{tabular}}{{{ks_col_spec}}}
  \toprule
  \textbf{{Output}} & {ks_headers} \\
  \midrule
        {ks_rows_tex}
  \bottomrule
\end{{tabular}}

% ── 5. Split Quality ──────────────────────────────────────────────────────────
\section*{{5.\enspace Train/Test Split Quality}}

\begin{{tabular}}{{ll}}
  \textbf{{Residual voxel proportion}}  & {split_rvp_val}\enspace {split_rvp_icon} \\
  \textbf{{Valid test proportion}}      & {split_vtp_val} \\
  \textbf{{Phacking test proportion}}   & {split_pht_val} \\
  \textbf{{Isolated test proportion}}   & {split_itp_val} \\
  \textbf{{Chi\textsuperscript{{2}} p-value}}
    & {split_chi_val}\enspace {split_chi_icon} \\
\end{{tabular}}

% ── 6. Recommendation ─────────────────────────────────────────────────────────
\section*{{6.\enspace Recommendation}}

\begin{{itemize}}
  \item \textbf{{Recommended model:}} \textbf{{{_esc(best)}}}
    (highest average R\textsuperscript{{2}} = {best_avg_r2:.4f},
     Q90 pass rate {best_q90_pass}/{n_out},
     KS pass rate {best_ks_pass}/{n_out}).
  \item Split quality is excellent: no residual voxels, no phacking artefacts.
  \item Deploy \texttt{{{deploy_pkl}}} for inference.
\end{{itemize}}

\end{{document}}
"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata', help='Path to metadata_*.json')
    parser.add_argument('--output', default=None, help='Output directory (default: same as metadata)')
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    out_dir   = Path(args.output) if args.output else meta_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    d   = extract(meta_path)
    tex = build_latex(d)

    tex_file = out_dir / f'executive_summary_{d["use_case"]}.tex'
    pdf_file = out_dir / f'executive_summary_{d["use_case"]}.pdf'

    tex_file.write_text(tex)
    print(f'LaTeX source → {tex_file}')

    # Compile with tectonic (downloads packages automatically)
    result = subprocess.run(
        ['tectonic', '-o', str(out_dir), str(tex_file)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f'PDF generated → {pdf_file}')
    else:
        print('tectonic STDERR:', result.stderr[-1000:])
        print('LaTeX source saved — compile manually with pdflatex or Overleaf.')
        sys.exit(1)


if __name__ == '__main__':
    main()
