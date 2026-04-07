# ML Training Debugger — OpenEnv

An RL environment where an AI agent debugs broken ML training runs.
The agent inspects training logs, metrics, and configurations to identify
root causes and apply correct fixes — mirroring real-world ML engineering workflows.

---

## Environment Description

Real ML training runs break in subtle ways: loss diverges, models overfit, data leaks
into validation sets, gradients vanish. This environment simulates those failures and
challenges an agent to reason across multiple signals to restore healthy training.

The agent interacts via a text-based action space, receiving structured observations
(logs, metrics, config, alerts) and must diagnose and fix the issue within a step budget.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| task_id | string | Current task identifier |
| step | int | Current step number |
| training_logs | list[str] | Recent training log lines |
| metrics_history | list[TrainingMetrics] | Per-epoch loss/accuracy/gradient metrics |
| current_config | TrainingConfig | Current training hyperparameter config |
| system_alerts | list[str] | Warnings and critical alerts |
| available_actions | list[str] | Valid action types |
| diagnosis_history | list[str] | Agent's previous diagnoses |
| fix_history | list[str] | Agent's previously applied fixes |
| is_training_healthy | bool | Whether current config resolves the issue |

## Action Space

| action_type | parameters | Description |
|---|---|---|
| inspect_logs | {} | Read training logs |
| inspect_metrics | {} | Examine epoch metrics |
| inspect_config | {} | View current config |
| check_data | {} | Inspect data pipeline |
| diagnose_issue | {"diagnosis": str} | Submit a root cause diagnosis |
| modify_config | {"key": str, "value": any} | Change a config parameter |
| apply_fix | {"fix_type": str, ...} | Apply a structural fix |
| restart_training | {} | Restart with current config |
| submit_diagnosis | {} | Finalize and trigger grading |

---

## Tasks

### easy_lr_divergence (Easy)
Training loss is diverging to infinity. The agent must identify the learning rate
is too high and reduce it to a stable value.
- Max steps: 10
- Expected score: 1.0 for correct diagnosis + fix in ≤3 steps

### medium_data_leakage (Medium)
Validation accuracy is unrealistically high and val_loss << train_loss.
The agent must detect data leakage and fix the train/val split.
- Max steps: 15
- Requires inspecting logs, metrics, and data pipeline

### hard_overfitting_cascade (Hard)
Severe overfitting: train_acc=99%, val_acc=41%. Multiple missing regularization
settings are contributing. Agent must apply a sequence of 3 correct fixes.
- Max steps: 20
- Partial credit per correct fix in sequence

---

## Reward Function

- +0.05 per signal inspected (logs, metrics, config, data)
- +0.20 for correct diagnosis
- -0.05 for incorrect diagnosis
- +0.15–0.30 per correct fix applied (scales with task difficulty)
- -0.10 for invalid actions
- Final graded score (0.0–1.0) on submit_diagnosis or episode end

---

## Setup & Usage

```bash
pip install -r requirements.txt
python app.py
```

API available at `http://localhost:7860`

### Docker

```bash
docker build -t ml-debug-env .
docker run -p 7860:7860 ml-debug-env
```

### Run Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_hf_token"
python inference.py
```

---

## Baseline Scores

| Task | Score | Difficulty |
|---|---|---|
| easy_lr_divergence | ~0.80 | Easy |
| medium_data_leakage | ~0.65 | Medium |
| hard_overfitting_cascade | ~0.45 | Hard |

---

## API Endpoints

- `POST /reset` — Start new episode `{"task_id": "easy_lr_divergence"}`
- `POST /step` — Take action `{"task_id": "...", "action": {"action_type": "...", "parameters": {}}}`
- `GET /state` — Get current state `?task_id=easy_lr_divergence`
- `POST /close` — Get final graded score
- `GET /tasks` — List all tasks
