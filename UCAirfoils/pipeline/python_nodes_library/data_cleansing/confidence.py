import pandas as pd
import surrogate_factory as sf

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
