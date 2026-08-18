import pandas as pd
import surrogate_factory as sf

# NeuralFoil's own names for the transition locations, which earlier runs wrote
# straight into the dataset. The pipeline now uses the spelled-out names, so a
# dataset generated before that change would fail in SF_5 with a KeyError on
# the new name. Renaming here means existing data keeps working without
# regenerating it.
LEGACY_NAMES = {
    'Top_Xtr': 'top_transition_location',
    'Bot_Xtr': 'bottom_transition_location',
}


@sf.node
def rename_legacy_outputs(workflow, input_table) -> pd.DataFrame:
    """Bring pre-rename datasets up to the current output names."""
    present = {old: new for old, new in LEGACY_NAMES.items()
               if old in input_table.columns and new not in input_table.columns}
    if not present:
        print("  Output names already current — nothing to rename.")
        return input_table

    print(f"  Renaming legacy columns: {present}")
    return input_table.rename(columns=present)

# NeuralFoil reports how much it trusts each evaluation. Random Kulfan weight
# combinations produce many implausible sections, and those come back with low
# confidence and visibly noisier coefficients — on a 20k sample the CD spread
# was 0.034 below this threshold against 0.013 above it. Training on them costs
# accuracy, so they are dropped here rather than filtered later.
CONFIDENCE_THRESHOLD = 0.5


@sf.node
def filter_by_confidence(workflow, input_table) -> pd.DataFrame:
    """Drop rows NeuralFoil could not analyse reliably."""
    col = 'analysis_confidence'
    if col not in input_table.columns:
        print(f"  '{col}' not present — nothing to filter.")
        return input_table

    keep = input_table[col] >= CONFIDENCE_THRESHOLD
    kept = input_table[keep].drop(columns=[col]).reset_index(drop=True)

    print(f"  threshold : {CONFIDENCE_THRESHOLD}")
    print(f"  kept      : {len(kept):,} of {len(input_table):,} ({keep.mean():.1%})")
    print(f"  dropped   : {(~keep).sum():,}")
    for c in ('CL', 'CD', 'CM'):
        if c in input_table.columns:
            print(f"    {c:<4} std  dropped {input_table.loc[~keep, c].std():.4f}"
                  f"   kept {input_table.loc[keep, c].std():.4f}")

    workflow.metadata.update_step_data({
        'confidence_filter': {
            'column': col,
            'threshold': CONFIDENCE_THRESHOLD,
            'rows_in': int(len(input_table)),
            'rows_out': int(len(kept)),
        }
    })
    return kept
