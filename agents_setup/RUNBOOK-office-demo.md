# Runbook: pipeline-builder demo on the office PC (local Python)

Demo case: **UCAirfoilNoise**, built on the NASA airfoil self-noise dataset (UCI Machine Learning Repository, dataset 291). It is a real public benchmark, so the agent's result can be contrasted against published numbers instead of trusting the agent's own report.

The dataset: 1,503 wind-tunnel measurements of NACA 0012 airfoil sections (NASA, anechoic wind tunnel). 5 inputs: frequency (Hz), angle of attack (deg), chord length (m), free-stream velocity (m/s), suction-side displacement thickness (m). 1 output: scaled sound pressure level (dB), range 103.3 to 140.9 (never crosses zero, so relative Q90 is safe). A semicolon-separated copy with headers is committed at `agents_setup/demo_data/airfoil_self_noise.csv`, so no internet is needed.

## Reference results to contrast (independent of the agent)

| Source | Model | R² (test) |
|---|---|---|
| Published (MDPI Eng. Proc. 2023, 70/30 split) | Random Forest | 0.929 |
| Published (same paper) | Gradient Boosting | 0.862 |
| Published (same paper, best) | Extra Trees | 0.948 |
| Our own verified run (80/20 split, seed 42, tuned GB) | GradientBoosting (500 trees, depth 5) | 0.954, MAE 0.98 dB, Q90 = 0.019 |
| Our own verified run (baseline) | LinearRegression | 0.558, MAE 3.67 dB, Q90 = 0.066 |

Acceptance band for the demo: champion R² between 0.86 and 0.96 and Q90 < 0.03 → consistent with the literature, PASS. Linear baseline must fail (the frequency-noise relation is strongly non-linear). If the agent reports R² > 0.99, suspect test-set leakage; if R² < 0.80, the model or split is wrong: both are useful talking points, not demo failures.

## 0. One-time Python setup (before the demo)

1. Install Python 3.11 (python.org installer or the company software portal; tick "Add python.exe to PATH" on Windows).
2. From the Pipelines workspace root, create the environment and install the pinned packages:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

(`pip install -e .` installs `surrogate_factory` from `src/`; run it from the repo root. On Mac/Linux the activation is `source .venv/bin/activate`.)

3. Verify:

```
python --version
python -c "import surrogate_factory; print('SF OK')"
```

4. In VS Code, select this interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → `.venv`.
5. Optional, only for the executive summary PDF: a LaTeX distribution (MiKTeX on Windows). Without it the demo still works; the PDF stage is skipped.

## 1. Install the agent

Copy `agents_setup/agents/pipeline-builder.agent.md` to `<workspace>\.github\agents\` and `agents_setup/skills/sf-pipeline/SKILL.md` to `<workspace>\.github\skills\sf-pipeline\`. Reload VS Code and check `pipeline-builder` appears in the Chat agent picker ("Configure Custom Agents..." as fallback). Make sure `agents_setup/demo_data/airfoil_self_noise.csv` is in the workspace.

## 2. The demo prompt

Select agent `pipeline-builder`, model `(DIV) GPT 120B - medium`, and paste:

```
Build the use case UCAirfoilNoise end to end from the dataset
agents_setup/demo_data/airfoil_self_noise.csv (semicolon separated, header row).
Inputs: frequency_hz, angle_of_attack_deg, chord_length_m, velocity_ms,
displacement_thickness_m. Output: spl_db. Requirement: Q90 < 0.03 relative
error on the test set. This is the public NASA airfoil self-noise dataset;
published tree-ensemble results reach R2 0.86-0.95, so compare your champion
against that range. Verify the Python environment first and show me the plan
before writing files.
```

If the agent proposes copying an existing UC folder, reject the step and tell it to re-read its instructions: every file must be generated for this dataset.

## 3. What a correct run looks like

1. Environment check passes; the agent states whether `surrogate_factory` is importable (native SF mode) or not (standalone-stages mode, same folder structure).
2. Plan: UCAirfoilNoise tree with `pipeline_config.yaml`, `metadata/SF_1..SF_9.yaml`, nodes, `run_pipeline.py`; champion (GradientBoosting or RandomForest) and baseline (LinearRegression) with one-line justifications. Approve it.
3. Files generated, smoke run on a subsample, then full run (a couple of minutes on 1,503 rows).
4. Final report: verdict table with Q90 vs 0.03, champion vs baseline, and the comparison against the published R² range.

## 4. Evidence to capture for the slides

| # | Acceptance check | Evidence |
|---|---|---|
| A | Environment verified; SF-import check reported | chat screenshot |
| B | Plan shown and approved; no copying of any existing UC | chat screenshot |
| C | UCAirfoilNoise tree with generated config + 9 stage YAMLs + run_pipeline.py | explorer screenshot |
| D | Run completes; verdict PASS with Q90 < 0.03 | terminal screenshot |
| E | Champion R² inside the published 0.86-0.95 band; linear baseline fails | chat screenshot + metrics.json |

E is the key slide: the agent's pipeline reproduces published results on a public NASA benchmark. That is external, checkable proof, not the agent grading its own homework.

## 5. Troubleshooting

- `python` not found in the chat terminal: the venv is not activated or the wrong interpreter is selected; redo step 0.4.
- `import surrogate_factory` fails: `pip install -e .` was not run from the repo root; run it inside the activated venv.
- Rate limit / 502 from DAISEI: wait a minute and resend; the agent resumes.
- Same command failing twice: the agent must stop and report by instruction; paste the verbatim error back to it.
- Agent cannot execute commands at all: it switches to prepare-only mode by instruction — it still generates the full use case plus a `RUN_ME.md` with the exact commands. Run them yourself in a terminal, paste the output back into the chat, and the agent continues with the verification and the verdict. The demo still works: "the agent built it, I only pressed run".

## 6. Variation for a second demo

Ask for a different public dataset with the same prompt shape (e.g. UCI Concrete Compressive Strength or Combined Cycle Power Plant) — the agent generates a different, adapted pipeline with the same SF structure. That, plus the benchmark match, is the proof it adapts instead of template-pasting.
