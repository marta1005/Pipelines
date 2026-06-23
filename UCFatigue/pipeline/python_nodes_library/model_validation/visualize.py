import pandas as pd
import surrogate_factory as sf
from pathlib import Path
from surrogate_factory.catalog.visualization import error


@sf.node
def plot(workflow, Test_set, model_output):
    """Generate error plots for each predicted output."""
    outdir = Path(workflow.config['artifacts.folder'])

    for mlmodels in workflow.metadata.get_step_data(
        ['metadata', 'Model_Training', 'Models']
    ):
        run_id = mlmodels['run_id']
        outputs = model_output.columns.tolist()
        inputs = [col for col in Test_set.columns if col not in outputs]

        for output in model_output.columns:
            y_pred = model_output[[output]]
            y_test = Test_set[inputs + [output]]

            outfile = outdir / f"{output}_{run_id}.pdf"
            outfile_png = outdir / f"{output}_{run_id}.png"

            error.create_plot(y_pred[output], y_test[output], outfile, outfile_png)

            input_table = pd.concat([
                Test_set[inputs].reset_index(drop=True),
                pd.DataFrame(y_pred[output].values, columns=["y_pred"]),
                pd.DataFrame(Test_set[output].values, columns=["y_test"]),
            ], axis=1)
            input_table['output'] = output
            error.plot_ratio(
                input_table, inputs, [output],
                outfile_png=outdir / f"ratio_{output}_{run_id}.png"
            )
            print(f"Plots saved for {output}")
