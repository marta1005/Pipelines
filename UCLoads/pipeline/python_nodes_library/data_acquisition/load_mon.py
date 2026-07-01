import numpy as np
import pandas as pd
import surrogate_factory as sf
from pathlib import Path


_RENAME = {
    # Inputs
    'Mach[-]':       'Mach',
    'Altitude[ft]':  'Altitude_ft',
    'ALPHA[deg]':    'ALPHA_deg',
    # WINGROOT outputs
    'Net-Fz[N]':     'Fz',
    'Net-Mx[Nm]':    'Mx',
    'Net-My[Nm]':    'My',
    # FUS outputs (same base names, disambiguated by suffix later)
    'Net-Fy[N]':     'Fy',
    'Net-Mz[Nm]':    'Mz',
}


def _clean_col(name: str) -> str:
    """Map raw .mon column name to a clean Python identifier."""
    return _RENAME.get(name, name)


def _parse_mon(filepath):
    """Read a Tecplot-style .mon file.

    Format:
        Line 0-2 : arbitrary header (ignored)
        Line 3   : VARIABLES="col1""col2"..."colN"
        Line 4+  : whitespace-separated data rows

    Returns (columns: list[str], data: np.ndarray).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    var_line = ''.join(lines[3].split())          # collapse whitespace
    var_line = var_line.replace('VARIABLES=', '')
    var_line = var_line.strip('"')                 # remove surrounding quotes
    raw_cols = var_line.split('""')                # split on double-double-quote
    columns = [_clean_col(c) for c in raw_cols]

    data_lines = lines[4:]
    data = np.array(
        [list(map(float, l.split())) for l in data_lines if l.strip()],
        dtype=np.float32,
    )
    return columns, data


@sf.node
def batch_extract(workflow, input_table=None):
    """Load all .mon files listed in metadata, merge outputs into one DataFrame.

    Each monitor station contributes its own output columns (suffixed with the
    station label to avoid name collisions).  All stations must share the same
    input rows in the same order.
    """
    metadata = workflow.get_step()
    data_folder = Path(metadata['data_folder'])
    input_cols = metadata['inputs']
    stations = metadata['monitor_stations']

    merged = None
    for station in stations:
        filepath = data_folder / station['file']
        columns, data = _parse_mon(filepath)
        df = pd.DataFrame(data.astype(np.float64), columns=columns)

        # Verify inputs are present
        for c in input_cols:
            if c not in df.columns:
                raise ValueError(f"{station['file']}: missing input column '{c}'")

        # Rename output columns with station suffix to avoid collisions
        suffix = station.get('suffix', '')
        out_cols = station['outputs']
        rename = {c: c + suffix for c in out_cols if c in df.columns}
        df = df.rename(columns=rename)

        keep = input_cols + [c + suffix for c in out_cols]
        df = df[keep]

        if merged is None:
            merged = df
        else:
            # Merge on inputs (inner join — rows must match)
            merged = merged.merge(
                df[[c + suffix for c in out_cols]],
                left_index=True,
                right_index=True,
            )
        print(f"  Loaded {filepath.name}: {df.shape[0]} rows, "
              f"outputs: {[c + suffix for c in out_cols]}")

    print(f"Dataset shape after merge: {merged.shape}")
    return merged


@sf.node
def batch_transform(workflow, parsed_batches):
    """Pass-through transform (placeholder for future ETL steps)."""
    return parsed_batches


@sf.node
def batch_load(workflow, transformed_batches, input_table=None):
    return transformed_batches


@sf.node
def data_visualization(workflow, input_table):
    try:
        from surrogate_factory.catalog.visualization.profiling import profiling_report
        profiling_report(input_table, "UCLoads Raw Data")
    except Exception:
        print("Profiling not available.")
