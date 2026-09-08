# Runbook: easy-pipeline demo over the SSH bridge to multivac

Context: the office PC has no Python. All execution happens on the multivac cluster (`mln4`) through SSH; the agent writes files locally and runs everything remotely. This supersedes the demo section of `RUNBOOK-office-setup-and-demo.md`.

## 0. One-time SSH setup (5 minutes, do this BEFORE the demo)

Goal: passwordless key-based login so the agent never handles a password. Do not put any password in any file; this repo is pushed to GitHub.

1. Generate a key on the office PC (press Enter at every prompt):

```
ssh-keygen -t ed25519
```

2. Install the public key on the cluster (you will type your password once, in the terminal only):

```
ssh c05279@mln4 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys" < ~/.ssh/id_ed25519.pub
```

3. Create `~/.ssh/config` on the office PC with exactly:

```
Host mln4
    HostName mln4
    User c05279
```

4. Test the bridge (must print a Python version with no password prompt):

```
ssh mln4 "source /home/FlightPhysicsValidation/flowsimTest/dev_env.sh && python --version"
```

5. Create the working folder on the cluster:

```
ssh mln4 "mkdir -p ~/agent_demo"
```

Security note: the account password was shared in plain text recently. Change it (`passwd` on mln4) after installing the key.

## 1. Install the agent

Copy `easy-pipeline.agent.md` to `<workspace>\.github\agents\easy-pipeline.agent.md`, reload VS Code, and check that `easy-pipeline` appears in the Chat agent picker (same procedure as the other agents; "Configure Custom Agents..." as fallback).

## 2. The demo prompt

Select agent `easy-pipeline`, model `(DIV) GPT 120B - medium`, and paste:

```
Build a small pipeline from scratch that learns the tip deflection of a
cantilever beam from synthetic data. Follow your default demo spec. Verify
the SSH bridge first, then show me the plan before writing files.
```

The agent must NOT copy any existing UC folder. If it proposes copying anything, reject the step and tell it to re-read its instructions.

## 3. What a correct run looks like

1. Bridge check: `ssh mln4 "source ... && python --version"` succeeds.
2. Plan: 5 files (`generate_data.py`, `train.py`, `validate.py`, `run_pipeline.py`, `README.md`), the physics formula, the Q90 < 0.05 requirement. Approve it.
3. Agent writes the files locally under `UCEasy/` (all generated, each under 80 lines).
4. Sync + smoke run (100 points), then full run (2,000 points):

```
scp -r UCEasy mln4:~/agent_demo/
ssh mln4 "source /home/FlightPhysicsValidation/flowsimTest/dev_env.sh && cd ~/agent_demo/UCEasy && python run_pipeline.py"
scp -r mln4:~/agent_demo/UCEasy/outputs UCEasy/outputs
```

5. Final report in chat: metrics table (GradientBoosting champion vs LinearRegression baseline: R2, MAE, Q90), `outputs/verdict.txt` with PASS, and `outputs/pred_vs_true.png`.

Expected result: GB passes Q90 < 0.05 comfortably; the linear baseline fails (the law is cubic in L, so a linear model cannot fit it). That contrast is the story: the agent built, ran, and validated the pipeline end to end, and the verdict is physically explainable.

## 4. Evidence to capture for the slides

| # | Acceptance check | Evidence |
|---|---|---|
| A | Bridge verified: `python --version` over SSH before anything else | chat screenshot |
| B | Plan shown and approved; no mention of copying an existing UC | chat screenshot |
| C | `UCEasy/` tree with the 5 generated files | explorer screenshot |
| D | Remote run completes; `outputs/metrics.json` + `verdict.txt` PASS | terminal screenshot |
| E | `pred_vs_true.png` plus the champion-vs-baseline interpretation | image + chat screenshot |

## 5. Troubleshooting

- Password prompt appears: the key install (step 0.2) did not work; redo it. Never type the password into the chat.
- `python: command not found`: the `source .../dev_env.sh` part is missing from the remote command; every remote call must include it (each SSH command is a fresh shell).
- `scp` fails on `outputs/`: the run did not produce outputs; check the remote run log first.
- Rate limit / 502 from DAISEI: wait a minute and resend; the agent resumes.
- Agent stalls repeating the same failing command: it must stop after 2 identical failures by instruction; if not, stop it and paste the exact error back.

## 6. Variations for a second demo

The agent's spec says "unless the user specifies another problem": ask for any other simple law (projectile range, RC discharge, ideal gas) and it will generate a different pipeline from scratch with the same 5-file structure. That is the proof it is not template-pasting.
