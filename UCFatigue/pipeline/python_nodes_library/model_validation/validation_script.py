import os
import sys
import json
import argparse
import tempfile

import nbformat
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

from bs4 import BeautifulSoup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths that must be importable inside the executed notebook
_REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "..", ".."))
_LIB_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def _make_temp_kernelspec():
    """
    Create a throw-away kernelspec pointing to the current Python executable.
    Returns (tmpdir, kernel_name) — caller is responsible for cleanup.
    This avoids any dependency on a named kernel being installed in the system.
    """
    kernel_name = "sf_validation_kernel"
    tmpdir = tempfile.mkdtemp()
    # jupyter_client looks in {JUPYTER_PATH}/kernels/{name}/kernel.json
    spec_dir = os.path.join(tmpdir, "kernels", kernel_name)
    os.makedirs(spec_dir)

    pythonpath = ":".join([
        os.path.join(_REPO_ROOT, "src"),  # surrogate_factory
        _REPO_ROOT,                        # validationlib
        _LIB_DIR,                          # python_nodes_library
    ])

    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "SF Validation",
        "language": "python",
        "env": {
            "PYTHONPATH": pythonpath,
            "MLFLOW_ALLOW_FILE_STORE": "true",
        },
    }
    with open(os.path.join(spec_dir, "kernel.json"), "w") as f:
        json.dump(spec, f)

    return tmpdir, kernel_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Validation Notebook and export it to HTML")
    parser.add_argument("-d", "--csv_dir", required=True, help="Directory containing the CSV files")
    parser.add_argument("-n", "--model_name", default=None, help="Name of the model to validate")
    parser.add_argument("-o", "--output_dir", default=None, help="Directory to save the HTML output")
    parser.add_argument("--exclude_input", action="store_true", help="Exclude code cells from the HTML output")
    parser.add_argument("--exclude_warnings", action="store_true", help="Exclude warnings from the HTML output")
    parser.add_argument("--subsample_size", type=int, default=None, help="Number of examples to subsample from each set")
    parser.add_argument("--splitting_variables", nargs="+", type=str, default=None, help="List of variables to use for splitting the data (default: all variables)")
    parser.add_argument("--save_as_notebook", action="store_true", help="Save the executed notebook instead of exporting to HTML")
    parser.add_argument("--omit_execution", action="store_true", help="Omit the execution of the notebook and only save a copy of the template with the configuration (useful for debugging)")

    args = parser.parse_args()

    # Load the notebook template (always from the same directory as this script)
    template_path = os.path.join(SCRIPT_DIR, "validation_template.ipynb")
    with open(template_path) as file:
        notebook = nbformat.read(file, as_version=4)

    # Inject configuration into the first (parameter) cell
    config_str = "csv_dir = '{}'".format(args.csv_dir)
    config_str += "\nmodel_name = '{}'".format(args.model_name)
    config_str += "\nsubsample_size = {}".format(args.subsample_size)
    config_str += "\nsplitting_variables = {}".format(args.splitting_variables)
    notebook["cells"][0]["source"] = notebook["cells"][0]["source"] + "\n\n" + config_str

    if not args.omit_execution:
        tmpdir, kernel_name = _make_temp_kernelspec()
        try:
            ep = ExecutePreprocessor(
                timeout=600,
                kernel_name=kernel_name,
                # Tell jupyter_client where to find our temporary kernelspec
                kernel_manager_class="jupyter_client.manager.KernelManager",
            )
            # Prepend the temp dir so our kernelspec is found first
            os.environ["JUPYTER_PATH"] = tmpdir + os.pathsep + os.environ.get("JUPYTER_PATH", "")
            resources = {"metadata": {"path": SCRIPT_DIR}}
            ep.preprocess(notebook, resources)
        except CellExecutionError as e:
            print(f"Error executing the notebook: {e}")
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            sys.exit(1)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    if args.output_dir is None:
        output_dir = os.path.join(args.csv_dir, "validation_output")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if not args.save_as_notebook:
        # Export the executed notebook to HTML
        html_exporter = HTMLExporter()
        html_exporter.exclude_input = args.exclude_input
        html_data, _ = html_exporter.from_notebook_node(
            notebook,
            resources={"metadata": {"name": args.model_name}}
        )

        if args.exclude_warnings:
            soup = BeautifulSoup(html_data, "html.parser")
            for warning in soup.find_all("div", {"data-mime-type": "application/vnd.jupyter.stderr"}):
                warning.decompose()
            html_data = str(soup)

        filename = (f"{args.model_name}_validation_output.html"
                    if args.model_name else "validation_output.html")
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(html_data)
        print(f"Report saved: {os.path.join(output_dir, filename)}")
    else:
        filename = (f"{args.model_name}_validation_notebook.ipynb"
                    if args.model_name else "validation_notebook.ipynb")
        with open(os.path.join(output_dir, filename), "wt", encoding="utf-8") as file:
            nbformat.write(notebook, file)
        print(f"Notebook saved: {os.path.join(output_dir, filename)}")
