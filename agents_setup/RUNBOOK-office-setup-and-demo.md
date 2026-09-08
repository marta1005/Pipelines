# Runbook: SF pipeline-builder agent on the office PC (no GitHub required)

The `.github/` directory is only a folder name that VS Code scans inside the open workspace. It works on a plain local folder: no git repository, no GitHub account, no network beyond the DAISEI endpoint.

## 0. Prerequisites (office PC)

- The Pipelines workspace on local disk (SF v2.2 in `src/`, `validationlib/`, `UCHardLanding/` as template, working `.venv`).
- VS Code with the Chat view configured for the DAISEI DIV models (as in the DAISEI team setup; model picker shows `(DIV) GPT 120B - medium`).
- The three files from `agents_setup/`: `skills/sf-pipeline/SKILL.md`, `agents/pipeline-builder.agent.md`, `agents/validation-interpreter.agent.md`. Bring them on a USB stick, email, or shared drive; they are plain text.

## 1. Install the skill and the agents (5 minutes)

1. In the workspace root, create the folders (File Explorer or terminal):

```
mkdir .github\skills\sf-pipeline
mkdir .github\agents
```

2. Copy the files to:

```
<workspace>\.github\skills\sf-pipeline\SKILL.md
<workspace>\.github\agents\pipeline-builder.agent.md
<workspace>\.github\agents\validation-interpreter.agent.md
```

3. Reload VS Code (`Ctrl+Shift+P` → "Developer: Reload Window").
4. Open Chat. In the mode/agent picker you should now see `pipeline-builder` and `validation-interpreter`. If they do not appear, use the same picker → "Configure Custom Agents..." → create a new workspace agent and paste the file content; VS Code then writes it in the location that build expects.
5. If skills are not picked up, check Settings → search "agent skills" → enable it (recent VS Code builds; see code.visualstudio.com/docs/agent-customization/agent-skills).

Wiring check (30 seconds, before the real demo):

```
@pipeline-builder Which conventions do you follow when building a Surrogate Factory pipeline?
```

A correct setup answers with the skill's specifics: UPPERCASE `job_name` + `_1`, 9 SF stages, `random_state: 42`, champion + baseline, smoke run first. If the answer is generic, the SKILL.md was not loaded: re-check the folder path and reload.

## 2. Prepare the demo dataset (reproducible on purpose)

Reuse the hard-landing data under a new use-case name, so the expected result is known in advance:

```
mkdir UCDemo\data
copy UCHardLanding\data\datos.csv UCDemo\data\datos.csv
```

## 3. Run the demo

Select agent `pipeline-builder`, model `(DIV) GPT 120B - medium`, and paste:

```
Build a new use case UCDemo from the dataset UCDemo/data/datos.csv (semicolon
separated). Inputs I1..I7, outputs O1 and O2, requirement Q90 < 0.10 per output.
Choose champion and baseline models with justification, scaffold the full SF v2.2
pipeline, do a smoke run, then the production run, and finish with the verdict
table and corrective actions.
```

Approve the file edits and terminal commands when prompted ("Allow in this Session"). The production GB run on 89 k rows takes a few minutes.

## 4. Evidence that it works (capture these for the slides)

| # | Acceptance check | Evidence |
|---|---|---|
| A | Agent proposes a plan naming SF_1..SF_9 and justifies GradientBoosting champion / MLP baseline | chat screenshot |
| B | `UCDemo/pipeline/` exists: `pipeline_config.yaml`, 9 `metadata/SF_*.yaml`, `python_nodes_library/`, `run_pipeline.py` | explorer screenshot |
| C | `run_pipeline.py` completes SF_1..SF_9; `mlruns/` and `data/artifacts/` populated | terminal screenshot |
| D | `validation_reports/executive_summary_UCDEMO_1.pdf` generated; verdict table O1 and O2 PASS with Q90 close to the UCHardLanding reference (O1 ≈ 0.042, O2 ≈ 0.090) | PDF page 1 |
| E | `@validation-interpreter Assess UCDemo` returns the 4 blocks (split, accuracy, bias, coverage) with no corrective action needed | chat screenshot |

Check D is the objective proof: same data, agent-built pipeline, reproduces the numbers of the hand-built, validated UCHardLanding pipeline.

## 5. Troubleshooting

- "Rate limit exceeded" (30 requests/min) or 502: wait a minute and resend; the agent resumes where it stopped.
- Long training blocks the chat: let the agent give you the command, run `python UCDemo/pipeline/run_pipeline.py` in the terminal yourself, and paste the tail of the output back into the chat.
- Weak plan or malformed YAML: switch to `(DIV) GPT 120B - high` and ask it to re-check the YAMLs against the skill.
- Agent tries to edit `src/` or `validationlib/`: it must refuse by instruction; if it proposes it, reject the edit and tell it to re-read its instructions.

## 6. Optional second demo (30 seconds, zero risk)

`@validation-interpreter` on an existing case:

```
Assess UCAirfoils. Folder: UCAirfoils/pipeline/data/artifacts/validation_reports
```

Expected: split OK, R² 0.97-0.999, Q90 failures explained as the near-zero relative-error artefact with the SF_1 metric change as the corrective action.
