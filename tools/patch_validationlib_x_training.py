#!/usr/bin/env python
"""
Fix validationlib's training_curves_plot UnboundLocalError in place.

Older validationlib assigns x_training/x_validation only when a metric has BOTH
a training and a validation curve; any model with training data alone (gradient
boosting, or anything without early stopping) dies with
"local variable 'x_training' referenced before assignment".

Run this with the SAME Python that shows the error (the Jupyter kernel's
interpreter, or plain `python` on the office machine after `git pull`):

    python tools/patch_validationlib_x_training.py

It locates the validationlib that Python actually imports, checks whether the
fix is already there (then it does nothing), and otherwise inserts the two
defaults right before the both-curves branch, keeping a .bak copy next to the
file. An explicit file may be given instead:

    python tools/patch_validationlib_x_training.py /path/to/validationlib/models/validation.py
"""
import re
import shutil
import sys
from pathlib import Path

MARKER = 'if training_metric is not None else None'
ANCHOR = re.compile(
    r'^(?P<indent>[ \t]+)#?.*\n?'
    r'(?P<line>(?P=indent)if training_metric is not None and '
    r'validation_metric is not None:)', re.M)

BLOCK = """\
{i}# Defaults for the cases where only one of the two curves is present.
{i}# Without these, a metric with training but no validation data (a model
{i}# trained without early stopping, say) never assigns x_training and
{i}# fails with UnboundLocalError on the plot call below.
{i}x_training = (np.arange(1, len(training_metric) + 1)
{i}              if training_metric is not None else None)
{i}x_validation = (np.arange(1, len(validation_metric) + 1)
{i}                if validation_metric is not None else None)

"""


def target_file():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    import validationlib.models.validation as v
    return Path(v.__file__)


def main():
    path = target_file()
    if not path.exists():
        sys.exit(f'no existe: {path}')
    src = path.read_text()

    if 'def training_curves_plot' not in src:
        sys.exit(f'{path} no contiene training_curves_plot — ruta equivocada')
    if MARKER in src:
        print(f'OK: {path}\n    ya tiene el fix de x_training — nada que hacer')
        return

    m = re.search(r'^([ \t]+)if training_metric is not None and '
                  r'validation_metric is not None:', src, re.M)
    if not m:
        sys.exit(f'{path}: no encuentro la rama de ambas curvas — '
                 f'versión inesperada, parchea a mano (ver commit 992cd2b)')

    indent = m.group(1)
    patched = src[:m.start()] + BLOCK.format(i=indent) + src[m.start():]

    compile(patched, str(path), 'exec')  # never write something that won't parse

    backup = path.with_suffix(path.suffix + '.bak')
    shutil.copy2(path, backup)
    path.write_text(patched)

    # Drop stale bytecode so the running interpreter picks the fix up.
    pycache = path.parent / '__pycache__'
    if pycache.exists():
        shutil.rmtree(pycache, ignore_errors=True)

    print(f'PARCHEADO: {path}\n    backup en {backup.name}\n'
          f'    reinicia el kernel de Jupyter para que lo cargue')


if __name__ == '__main__':
    main()
