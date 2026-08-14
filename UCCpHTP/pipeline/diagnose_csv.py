"""
Report exactly what the pre-split CSVs contain, so a loading problem can be
diagnosed from facts rather than guesses.

    python diagnose_csv.py                 # uses data_split_folder from the config
    python diagnose_csv.py /path/to/data   # or point it at a folder directly
"""

import sys
from pathlib import Path

import pandas as pd

FILES = ['x_train.csv', 'x_val.csv', 'x_test.csv',
         'yt_train.csv', 'yt_val.csv', 'yt_test.csv']


def folder_from_argv_or_config():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    import yaml
    cfg = yaml.safe_load((Path(__file__).parent / 'pipeline_config.yaml').read_text())
    return Path(cfg['data_split_folder'])


def report(path: Path):
    print(f"\n{'=' * 70}\n{path.name}   ({path.stat().st_size / 1e6:.1f} MB)\n{'=' * 70}")

    raw = path.open('rb').read(400)
    if raw.startswith(b'\xef\xbb\xbf'):
        print("  !! file starts with a UTF-8 BOM — the first column name will carry \\ufeff")

    with path.open(encoding='utf-8-sig') as fh:
        lines = [fh.readline().rstrip('\r\n') for _ in range(3)]

    print("\n  first 3 lines, exactly as stored:")
    for i, line in enumerate(lines):
        if line:
            print(f"    [{i}] {line[:110]!r}")

    first = lines[0]
    print("\n  separator candidates on line 0:")
    for s, name in ((';', 'semicolon'), (',', 'comma'), ('\t', 'tab'),
                    ('|', 'pipe'), (' ', 'space')):
        n = first.count(s)
        if n:
            print(f"    {name:<10} x{n:<4} -> would give {n + 1} columns")

    def numeric(tok):
        try:
            float(tok.strip())
            return True
        except ValueError:
            return False

    print("\n  how pandas reads it, per separator:")
    for sep, name in ((';', "';'"), (',', "','"), ('\t', "'\\t'")):
        try:
            df = pd.read_csv(path, sep=sep, nrows=5)
            cols = list(df.columns)
            all_num = all(numeric(str(c)) for c in cols)
            verdict = ("header row looks like DATA -> pass header=None"
                       if all_num else "header row looks like NAMES")
            print(f"    sep={name:<5} {len(cols)} column(s): {cols[:8]}")
            print(f"    {'':<11}{verdict}")
        except Exception as e:
            print(f"    sep={name:<5} failed: {type(e).__name__}: {str(e)[:60]}")

    # What the pipeline's own loader decides
    print("\n  -> what SF_2 will do:")
    counts = {s: first.count(s) for s in (';', ',', '\t', '|')}
    sep = max(counts, key=counts.get)
    if counts[sep] == 0:
        sep = ','
    fields = [f for f in first.split(sep) if f.strip() != '']
    header = None if fields and all(numeric(f) for f in fields) else 0
    df = pd.read_csv(path, sep=sep, header=header, nrows=5)
    print(f"     separator {sep!r}, header={'row 0' if header == 0 else 'none'}, "
          f"{len(df.columns)} column(s)")
    print(f"     dtypes: {dict(df.dtypes.astype(str))}")


def main():
    folder = folder_from_argv_or_config()
    print(f"Looking in: {folder}")
    if not folder.is_dir():
        print("  that folder does not exist")
        return
    present = [f for f in FILES if (folder / f).exists()]
    missing = [f for f in FILES if not (folder / f).exists()]
    print(f"  found  : {present}")
    if missing:
        print(f"  MISSING: {missing}")
    print(f"  other files in the folder: {sorted(p.name for p in folder.iterdir())[:15]}")
    for f in present:
        report(folder / f)


if __name__ == '__main__':
    main()
