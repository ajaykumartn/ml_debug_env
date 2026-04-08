---
title: ML Training Debugger
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - ml-debugging
  - real-world
  - rl-environment
---

# ML Training Debugger — OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/openenv)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)

An RL environment where an AI agent debugs broken ML training runs. The agent inspects training logs, metrics, and configurations to identify root causes and apply correct fixes — mirroring real-world ML engineering workflows used daily at AI companies.

---

## Motivation

ML training failures cost the industry millions of hours annually. A single misconfigured learning rate, a data leakage bug, or missing regularization can waste days of GPU compute and delay model releases. Today, every ML engineer at every AI company debugs these failures manually — reading logs, cross-referencing metrics, and applying fixes through trial and error.

This environment turns that real debugging workflow into a structured RL benchmark. An agent trained here learns to:

- Systematically inspect multiple signals before acting (logs, metrics, config, data pipeline)
- Reason across correlated symptoms to identify root causes
- Apply targeted fixes rather than random hyperparameter searches
- Handle multi-cause failures where symptoms overlap and mislead

Unlike toy environments, every scenario in this benchmark maps directly to failures that engineers at Meta, Google, and Hugging Face encounter in production. An agent that scores well here is an agent that could genuinely assist in real MLOps workflows — reducing debugging time from hours to seconds.

The environment is designed to be unsolvable by random action (lazy agent scores 0.10 on hard tasks) but tractable for capable reasoning models (perfect agent scores 0.95–1.0), making it a meaningful benchmark for evaluating LLM reasoning quality in technical domains.

---

## Environment Overview

The agent starts with a broken training run and must:
1. Inspect available signals (logs, metrics, config, data pipeline)
2. Reason across multiple signals to identify the root cause
3. Apply the correct fix or sequence of fixes
4. Submit a final diagnosis

Each episode is randomized (with optional seed for reproducibility), so the agent cannot memorize solutions.

---

## Tasks

| Task ID | Difficulty | Description |
|---|---|---|
| `easy_lr_divergence` | Easy | Training loss diverging to infinity — identify and fix the learning rate |
| `medium_wrong_loss` | Medium | Accuracy stuck near random baseline — wrong loss function for classification |
| `medium_data_leakage` | Medium | Val metrics unrealistically good — detect and fix data leakage in the pipeline |
| `hard_overfitting_cascade` | Hard | Severe overfitting from 3 simultaneous missing regularization settings |
| `hard_dual_bug` | Hard | Two bugs active simultaneously — overfitting + learning rate instability |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Current task identifier |
| `step` | int | Current step in the episode |
| `training_logs` | list[str] | Training log lines (randomized per reset) |
| `metrics_history` | list[TrainingMetrics] | Per-epoch loss, accuracy, gradient norms |
| `current_config` | TrainingConfig | Current hyperparameter configuration |
| `system_alerts` | list[str] | Critical warnings and error alerts |
| `available_actions` | list[str] | Valid action types for this step |
| `diagnosis_history` | list[str] | Agent's previous diagnoses this episode |
| `fix_history` | list[str] | Agent's previously applied fixes |
| `is_training_healthy` | bool | Whether current config resolves the issue |

---

## Action Space

| Action | Parameters | Description |
|---|---|---|
| `inspect_logs` | `{}` | Read training log lines |
| `inspect_metrics` | `{}` | Examine epoch-level metrics |
| `inspect_config` | `{}` | View current hyperparameter config |
| `check_data` | `{}` | Inspect data pipeline and splits |
| `diagnose_issue` | `{"diagnosis": str}` | Submit root cause diagnosis |
| `modify_config` | `{"key": str, "value": any}` | Change a hyperparameter |
| `apply_fix` | `{"fix_type": str, ...}` | Apply a structural fix (e.g. data split) |
| `restart_training` | `{}` | Restart with current config |
| `submit_diagnosis` | `{}` | Finalize episode and trigger grading |

Valid config keys: `learning_rate`, `batch_size`, `optimizer`, `loss_function`, `epochs`, `dropout_rate`, `weight_decay`, `gradient_clip`, `scheduler`

Valid diagnoses: `learning_rate_too_high`, `learning_rate_too_low`, `wrong_loss_function`, `data_leakage`, `vanishing_gradient`, `exploding_gradient_optimizer_mismatch`, `overfitting_cascade`, `underfitting`, `batch_size_issue`, `scheduler_misconfiguration`

---

## Reward Function

Rewards are provided at every step — not just at episode end.

| Event | Reward |
|---|---|
| Inspecting a signal (logs/metrics/config/data) | +0.05 |
| Correct diagnosis | +0.20 |
| Incorrect diagnosis | -0.05 |
| Correct fix applied (easy) | +0.30 |
| Correct fix in sequence (hard) | +0.15 per fix |
| Invalid action | -0.10 |
| Final graded score on submit | 0.0 – 1.0 |

---

## Grading

Each task uses a deterministic grader that scores 0.0–1.0:

**Easy tasks** (0.4 diagnosis + 0.4 fix + 0.2 efficiency)

**Medium tasks** (0.3 diagnosis + 0.4 fix + 0.2 signal inspection + 0.1 efficiency)

**Hard tasks** (0.25 diagnosis + 0.45 fix sequence + 0.2 signal inspection + 0.1 efficiency)

Partial credit is awarded per correct fix in a sequence. Value matching uses 5% tolerance to handle float precision.

---

## Baseline Scores

Scores from `inference.py` with `seed=42` using `meta-llama/Llama-3.1-8B-Instruct`:

| Task | Score | Notes |
|---|---|---|
| easy_lr_divergence | ~0.40 | 8B model partially solves |
| medium_wrong_loss | ~0.20 | Requires stronger reasoning |
| medium_data_leakage | ~0.20 | Requires stronger reasoning |
| hard_overfitting_cascade | ~0.20 | Requires frontier model |
| hard_dual_bug | ~0.20 | Requires frontier model |

A frontier model (70B+) is expected to score 0.80–1.0 on easy/medium tasks and 0.60–0.95 on hard tasks. The environment is intentionally designed to challenge capable models while being unsolvable by random action.

---

## Project Structure

```
ml-training-debugger/
├── environment/
│   ├── __init__.py       # exports MLDebugEnv
│   ├── models.py         # Pydantic: Observation, Action, Reward, TrainingConfig
│   ├── simulator.py      # generates randomized broken training scenarios
│   ├── graders.py        # deterministic graders for all 5 tasks
│   ├── tasks.py          # task definitions and ground truth
│   └── env.py            # OpenEnv class: reset/step/state/close
├── app.py                # FastAPI server (port 7860)
├── inference.py          # LLM agent baseline
├── openenv.yaml          # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### Local

```bash
pip install -r requirements.txt
python app.py
# Server runs at http://localhost:7860
```

### Docker

```bash
docker build -t ml-training-debugger .
docker run -p 7860:7860 ml-training-debugger
```

### API Endpoints

```
GET  /health              liveness probe
GET  /                    environment info
GET  /tasks               list all tasks
POST /reset               start new episode  {"task_id": "easy_lr_divergence", "seed": 42}
POST /step                take action        {"task_id": "...", "action": {"action_type": "...", "parameters": {}}}
GET  /state               current state      ?task_id=easy_lr_divergence
POST /close               get final score    {"task_id": "..."}
```

### Run Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_hf_token"

python inference.py
```

Output format:
```
[START] task=easy_lr_divergence env=ml-training-debugger model=meta-llama/Llama-3.1-8B-Instruct
[STEP]  step=1 action=inspect_logs reward=0.05 done=false error=null
[STEP]  step=2 action=diagnose_issue reward=0.20 done=false error=null
[STEP]  step=3 action=modify_config reward=0.30 done=false error=null
[STEP]  step=4 action=submit_diagnosis reward=0.90 done=true error=null
[END]   success=true steps=4 score=0.90 rewards=0.05,0.20,0.30,0.90
```

---

## Reproducibility

Pass a `seed` to `reset()` for fully reproducible episodes:

```python
obs = env.reset(seed=42)   # identical scenario every time
obs = env.reset()          # randomized scenario
```

The inference baseline uses `seed=42` for all tasks.

---

## Real-World Relevance

This environment models debugging workflows that ML engineers perform daily:

- Diagnosing diverging training runs
- Detecting data contamination in evaluation pipelines
- Identifying overfitting from regularization gaps
- Debugging multi-cause failures with overlapping symptoms

An agent trained on this environment develops systematic debugging strategies directly applicable to real ML infrastructure.

---

## Team

TriStack — OpenEnv Hackathon 2025
