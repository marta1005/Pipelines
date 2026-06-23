from pathlib import Path
import surrogate_factory as sf
import pandas as pd


@sf.node
def plot(workflow, Test_set, model_output):
    """_summary_
    Plots model results and save the figures
    Args:
        workflow (_type_): Workflow class
        input_table (_type_): input_table
    """

    from surrogate_factory.catalog.visualization import error
    
    for mlmodels in workflow.metadata.get_step_data(['metadata', 'Model_Training', 'Models']):  #algorithms
        
        run_id = mlmodels['run_id']
            
        outputs = model_output.columns
        inputs  = [col for col in Test_set.columns if col not in outputs]
        for output in model_output.columns:

            y_pred = model_output[[output]]
            y_test = Test_set[inputs + [output]]
    
            outdir = Path(workflow.config['artifacts.folder'])
            outfile = outdir / (output + f"_{run_id}" + ".pdf")
            outfile_png = outdir / (output + f"_{run_id}" + ".png")
    
            fig1 = error.create_plot(y_pred[output], y_test[output], outfile, outfile_png)
            # fig1.show()
            input_table = pd.concat([Test_set[inputs], pd.DataFrame(y_pred[output].values, columns= ["y_pred"]), pd.DataFrame(Test_set[output].values, columns= ["y_test"])], axis=1)
            input_table['output'] = output
            fig2 = error.plot_ratio(input_table, inputs, [output], outfile_png=outdir / f"ratio_{output}_{run_id}.png")
