"""
Design of Experiments over the airfoil design space, evaluated with NeuralFoil.

define_doe   Sobol sample of the bounded variables, plus the derived and fixed
             ones, filtered by the design-space constraints.
launch_sim   Runs NeuralFoil on the DoE and appends CL, CD, CM and the two
             transition locations.
"""

import numpy as np
import pandas as pd
import surrogate_factory as sf


def _doe_cfg(workflow):
    return workflow.metadata.get_step_data(
        ['metadata', 'Data_Acquisition_Generation', 'Data_Generation',
         'Create_Design_of_Experiments']
    )


def _classify(inputs):
    """Split the design-space entries into sampled, derived and fixed."""
    sampled = [v for v in inputs if 'bounds' in v]
    derived = [v for v in inputs if 'equation' in v]
    fixed   = [v for v in inputs if 'value' in v]
    return sampled, derived, fixed


@sf.node
def define_doe(workflow) -> pd.DataFrame:
    """Sobol DoE over the design space, honouring bounds and constraints."""
    from scipy.stats import qmc

    cfg    = _doe_cfg(workflow)
    space  = cfg['Design_Space']
    doe    = cfg['DoE']
    size   = int(doe['size'])
    margin = float(doe.get('margin_on_bounds', 0) or 0)

    sampled, derived, fixed = _classify(space['inputs'])
    constraints = space.get('constraints') or []

    lo, hi, names, log_mask = [], [], [], []
    for v in sampled:
        a, b = (float(x) for x in v['bounds'])
        if margin:                      # shrink the box symmetrically
            span = b - a
            a, b = a + margin * span, b - margin * span
        use_log = str(v.get('scale', 'linear')).lower() == 'log'
        if use_log and a <= 0:
            raise ValueError(f"{v['name']}: log scale needs positive bounds, got {v['bounds']}")
        lo.append(np.log10(a) if use_log else a)
        hi.append(np.log10(b) if use_log else b)
        names.append(v['name'])
        log_mask.append(use_log)

    lo, hi = np.array(lo), np.array(hi)

    def draw(n):
        # Sobol is balanced in powers of two; take the next one up and trim.
        m = max(1, int(np.ceil(np.log2(max(n, 2)))))
        u = qmc.Sobol(d=len(names), scramble=True, seed=42).random_base2(m)[:n]
        x = qmc.scale(u, lo, hi)
        for j, is_log in enumerate(log_mask):
            if is_log:
                x[:, j] = 10.0 ** x[:, j]
        return pd.DataFrame(x, columns=names)

    def finish(df):
        """Add derived and fixed columns, then apply the constraints."""
        for v in derived:
            df[v['name']] = df.eval(v['equation'])
        for v in fixed:
            df[v['name']] = float(v['value'])
        for expr in constraints:
            df = df[df.eval(expr)]
        return df

    # Constraints reject part of the box, so oversample and top up if short.
    df = finish(draw(size))
    attempts = 0
    while len(df) < size and attempts < 5:
        attempts += 1
        deficit = size - len(df)
        rate = max(len(df) / size, 0.05)
        df = pd.concat([df, finish(draw(int(deficit / rate) + 1024))], ignore_index=True)
    df = df.head(size).reset_index(drop=True)

    order = [v['name'] for v in space['inputs']]
    df = df[[c for c in order if c in df.columns]]

    print(f"  DoE       : {doe['algorithm']} ({doe['tool']})")
    print(f"  sampled   : {len(names)} variables" +
          (f"  [log: {[n for n, l in zip(names, log_mask) if l]}]" if any(log_mask) else ""))
    print(f"  derived   : {[v['name'] for v in derived]}")
    fixed_desc = ', '.join('{}={}'.format(v['name'], v['value']) for v in fixed)
    print(f"  fixed     : {fixed_desc}")
    print(f"  constraints: {constraints}")
    print(f"  points    : {len(df):,} of {size:,} requested")

    workflow.metadata.update_step_data(
        {'generated_points': int(len(df)), 'variables': list(df.columns)},
        ['metadata', 'Data_Acquisition_Generation', 'Data_Generation',
         'Create_Design_of_Experiments', 'DoE'],
    )
    return df


@sf.node
def launch_sim(workflow, doe: pd.DataFrame, batch_size: int = 20000) -> pd.DataFrame:
    """
    Evaluate the DoE with NeuralFoil.

    Returns the design space plus CL, CD, CM, Top_Xtr, Bot_Xtr and
    analysis_confidence (NeuralFoil's own trust score for each point).
    """
    import neuralfoil as nf

    upper = [f'uW{i}' for i in range(1, 9)]
    lower = [f'lW{i}' for i in range(1, 9)]
    OUTPUTS = ['CL', 'CD', 'CM', 'Top_Xtr', 'Bot_Xtr', 'analysis_confidence']

    missing = [c for c in upper + lower + ['LE_weight', 'TE_thickness', 'alpha', 'reynolds']
               if c not in doe.columns]
    if missing:
        raise ValueError(f"DoE is missing {missing} — got {list(doe.columns)}")

    chunks = []
    for start in range(0, len(doe), batch_size):
        part = doe.iloc[start:start + batch_size]
        aero = nf.get_aero_from_kulfan_parameters(
            # NeuralFoil indexes the weights as upper_weights[i] for i in
            # range(8), so the arrays go in as (8, n_points) — transposed
            # relative to the DataFrame's (n_points, 8).
            kulfan_parameters=dict(
                upper_weights=part[upper].to_numpy().T,
                lower_weights=part[lower].to_numpy().T,
                leading_edge_weight=part['LE_weight'].to_numpy(),
                TE_thickness=part['TE_thickness'].to_numpy(),
            ),
            # Mach is fixed in the design space but NeuralFoil is incompressible,
            # so it is carried through as a column rather than passed here.
            alpha=part['alpha'].to_numpy(),
            Re=part['reynolds'].to_numpy(),
        )
        chunks.append(pd.DataFrame({k: np.asarray(aero[k]).ravel() for k in OUTPUTS},
                                   index=part.index))
        print(f"  evaluated {min(start + batch_size, len(doe)):>8,} / {len(doe):,}")

    out = pd.concat([doe, pd.concat(chunks)], axis=1)

    conf = out['analysis_confidence']
    print(f"\n  analysis_confidence: mean {conf.mean():.3f}   "
          f"below 0.5: {(conf < 0.5).sum():,} ({(conf < 0.5).mean():.1%})")
    for c in ('CL', 'CD', 'CM', 'Top_Xtr', 'Bot_Xtr'):
        print(f"    {c:<10} min {out[c].min():>10.4f}   max {out[c].max():>10.4f}")

    return out
