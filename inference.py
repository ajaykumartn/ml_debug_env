"""
Inference Script — ML Training Debugger (OpenEnv)
===================================================
Runs an LLM agent against all 3 tasks and emits structured stdout logs.

Required env vars:
  API_BASE_URL     LLM API endpoint
  MODEL_NAME       Model identifier
  HF_TOKEN         Hugging Face / API key

Stdout format (mandatory):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import sys
from openai import OpenAI
from environment import MLDebugEnv
from environment.models import Action
from environment.tasks import TASKS

# ── Config ─────────────────────────────────────────────────────────────────────
# Defaults set only for API_BASE_URL and MODEL_NAME (not HF_TOKEN) — per hackathon spec
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional, if using from_docker_image()

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

ENV_NAME = "ml-training-debugger"

SYSTEM_PROMPT = """You are an ML engineer. Debug the broken training run by following these steps IN ORDER:

Step 1: {"action_type": "inspect_logs", "parameters": {}}
Step 2: {"action_type": "inspect_metrics", "parameters": {}}
Step 3: {"action_type": "inspect_config", "parameters": {}}
Step 4: {"action_type": "check_data", "parameters": {}}
Step 5: {"action_type": "diagnose_issue", "parameters": {"diagnosis": "PICK_ONE"}}
Step 6: Apply fix with modify_config or apply_fix
Step 7: {"action_type": "submit_diagnosis", "parameters": {}}

DIAGNOSES - pick the one that matches the alerts:
- "learning_rate_too_high" → loss is NaN/Inf, diverging, gradient_norm > 100
- "wrong_loss_function" → accuracy stuck near random baseline, MSE on classification
- "data_leakage" → val_loss much lower than train_loss, val_accuracy unrealistically high
- "overfitting_cascade" → train_acc >> val_acc, val_loss increasing, no dropout/weight_decay
- "vanishing_gradient" → gradient_norm near 0, loss not moving

FIX ACTIONS:
- {"action_type": "modify_config", "parameters": {"key": "learning_rate", "value": 0.001}}
- {"action_type": "modify_config", "parameters": {"key": "loss_function", "value": "cross_entropy"}}
- {"action_type": "modify_config", "parameters": {"key": "dropout_rate", "value": 0.3}}
- {"action_type": "modify_config", "parameters": {"key": "weight_decay", "value": 0.001}}
- {"action_type": "modify_config", "parameters": {"key": "epochs", "value": 20}}
- {"action_type": "apply_fix", "parameters": {"fix_type": "fix_data_split", "train": 0.8, "val": 0.2}}

IMPORTANT: Reply with ONLY a JSON object. No explanation. No text. Just JSON."""


def build_user_prompt(obs: dict) -> str:
    metrics = obs['metrics_history'][-5:]
    # Summarize metrics for clarity
    metric_summary = []
    for m in metrics:
        line = f"  epoch={m['epoch']} train_loss={m['train_loss']}"
        if m.get('val_loss') is not None:
            line += f" val_loss={m['val_loss']}"
        if m.get('train_accuracy') is not None:
            line += f" train_acc={m['train_accuracy']}"
        if m.get('val_accuracy') is not None:
            line += f" val_acc={m['val_accuracy']}"
        if m.get('gradient_norm') is not None:
            line += f" grad_norm={m['gradient_norm']}"
        metric_summary.append(line)

    already_done = []
    if obs['diagnosis_history']:
        already_done.append(f"Diagnosed: {obs['diagnosis_history'][-1]}")
    if obs['fix_history']:
        already_done.append(f"Fixed: {', '.join(obs['fix_history'])}")

    # Tell agent what to do next based on progress
    if obs['diagnosis_history'] and not obs['fix_history']:
        next_step = f"\nYou have diagnosed: {obs['diagnosis_history'][-1]}. Now apply the fix using modify_config or apply_fix, then call submit_diagnosis."
    elif obs['diagnosis_history'] and obs['fix_history']:
        next_step = "\nFix applied. Call submit_diagnosis to finalize."
    else:
        next_step = ""

    return f"""=== ML DEBUG SESSION | Task: {obs['task_id']} | Step: {obs['step']} ===

TRAINING LOGS:
{chr(10).join(obs['training_logs'])}

SYSTEM ALERTS:
{chr(10).join(obs['system_alerts'])}

CURRENT CONFIG:
  learning_rate={obs['current_config']['learning_rate']}
  optimizer={obs['current_config']['optimizer']}
  loss_function={obs['current_config']['loss_function']}
  epochs={obs['current_config']['epochs']}
  dropout_rate={obs['current_config']['dropout_rate']}
  weight_decay={obs['current_config']['weight_decay']}
  gradient_clip={obs['current_config']['gradient_clip']}
  data_split={obs['current_config'].get('data_split')}

METRICS (recent epochs):
{chr(10).join(metric_summary) if metric_summary else '  No metrics yet'}

PROGRESS SO FAR:
{chr(10).join(already_done) if already_done else '  Nothing done yet'}
is_training_healthy: {obs['is_training_healthy']}{next_step}

What is your next action? Reply with JSON only:"""


def call_llm(messages: list, retries: int = 2) -> dict:
    """Call LLM and parse JSON action. Retries on parse failure."""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=128,
                temperature=0.1,
                stream=False,
            )
            content = response.choices[0].message.content.strip()

            # Strip markdown code fences
            if "```" in content:
                parts = content.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        content = part
                        break

            # Find first JSON object in response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                content = content[start:end]

            parsed = json.loads(content)
            if "action_type" in parsed:
                # Normalize diagnosis strings — LLMs often return shortened versions
                if parsed.get("action_type") == "diagnose_issue":
                    diag = parsed.get("parameters", {}).get("diagnosis", "")
                    valid = {
                        "learning_rate_too_high", "learning_rate_too_low",
                        "wrong_loss_function", "data_leakage", "vanishing_gradient",
                        "exploding_gradient_optimizer_mismatch", "overfitting_cascade",
                        "underfitting", "batch_size_issue", "scheduler_misconfiguration"
                    }
                    if diag not in valid:
                        # Map common LLM shorthand to valid strings
                        mapping = {
                            "overfitting": "overfitting_cascade",
                            "overfit": "overfitting_cascade",
                            "learning_rate": "learning_rate_too_high",
                            "high_learning_rate": "learning_rate_too_high",
                            "lr_too_high": "learning_rate_too_high",
                            "wrong_loss": "wrong_loss_function",
                            "loss_function": "wrong_loss_function",
                            "mse_loss": "wrong_loss_function",
                            "data_leak": "data_leakage",
                            "leakage": "data_leakage",
                            "vanishing": "vanishing_gradient",
                            "gradient_vanishing": "vanishing_gradient",
                            "exploding_gradient": "exploding_gradient_optimizer_mismatch",
                        }
                        normalized = mapping.get(diag.lower().replace(" ", "_"))
                        if normalized:
                            parsed["parameters"]["diagnosis"] = normalized
                        # else let env handle the invalid diagnosis
                return parsed
        except Exception:
            pass

    # Progressive fallback — advance through workflow based on step number
    # Count steps from message history
    step_num = sum(1 for m in messages if m["role"] == "user")
    if step_num <= 1:
        return {"action_type": "inspect_logs", "parameters": {}}
    elif step_num == 2:
        return {"action_type": "inspect_metrics", "parameters": {}}
    elif step_num == 3:
        return {"action_type": "inspect_config", "parameters": {}}
    elif step_num == 4:
        return {"action_type": "check_data", "parameters": {}}
    else:
        # Force a diagnosis attempt based on task context from alerts
        last_obs = messages[-1]["content"] if messages else ""
        if "NaN" in last_obs or "diverging" in last_obs or "learning_rate" in last_obs.lower():
            return {"action_type": "diagnose_issue",
                    "parameters": {"diagnosis": "learning_rate_too_high"}}
        elif "overlap" in last_obs or "leakage" in last_obs or "contamination" in last_obs:
            return {"action_type": "diagnose_issue",
                    "parameters": {"diagnosis": "data_leakage"}}
        elif "overfitting" in last_obs or ("train_acc" in last_obs and "val_acc" in last_obs):
            return {"action_type": "diagnose_issue",
                    "parameters": {"diagnosis": "overfitting_cascade"}}
        elif "MSE" in last_obs or "random baseline" in last_obs or "accuracy stuck" in last_obs:
            return {"action_type": "diagnose_issue",
                    "parameters": {"diagnosis": "wrong_loss_function"}}
        elif "gradient_norm" in last_obs and "0.000" in last_obs:
            return {"action_type": "diagnose_issue",
                    "parameters": {"diagnosis": "vanishing_gradient"}}
        else:
            return {"action_type": "submit_diagnosis", "parameters": {}}


def run_task(task_id: str, seed: int = 42) -> dict:
    """Run one full episode for a task. Seed ensures reproducible baseline."""
    env = MLDebugEnv(task_id=task_id)
    task_info = TASKS[task_id]
    obs = env.reset(seed=seed)
    obs_dict = obs.model_dump()

    rewards = []
    steps = 0
    done = False
    score = 0.0
    success = False

    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while not done and steps < task_info["max_steps"]:
            user_msg = build_user_prompt(obs_dict)
            messages.append({"role": "user", "content": user_msg})

            action_dict = call_llm(messages)
            messages.append({"role": "assistant", "content": json.dumps(action_dict)})

            action = Action(
                action_type=action_dict.get("action_type", "inspect_logs"),
                parameters=action_dict.get("parameters", {}),
            )

            try:
                obs, reward, done, info = env.step(action)
                obs_dict = obs.model_dump()
                rewards.append(reward.value)
                last_error = info.get("last_action_error")
                steps += 1

                error_str = last_error if last_error else "null"
                print(
                    f"[STEP]  step={steps} action={action.action_type} "
                    f"reward={reward.value:.2f} done={str(done).lower()} error={error_str}",
                    flush=True,
                )
            except RuntimeError as e:
                steps += 1
                print(
                    f"[STEP]  step={steps} action={action.action_type} "
                    f"reward=0.00 done=true error={str(e)}",
                    flush=True,
                )
                done = True
                break

        final_result = env.close()
        score = final_result.get("score", 0.0)
        # Clamp to [0, 1] as per sample script
        score = min(max(score, 0.0), 1.0)
        success = final_result.get("passed", False)

    except Exception as e:
        # Ensure [END] always emits even on exception
        try:
            env.close()
        except Exception:
            pass
        score = 0.0
        success = False
        if not rewards:
            rewards = [0.0]

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END]   success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return {"task_id": task_id, "score": score, "success": success, "steps": steps}


def main():
    results = []
    for task_id in TASKS.keys():
        result = run_task(task_id, seed=42)  # fixed seed = reproducible baseline
        results.append(result)
        print("", flush=True)  # blank line between tasks

    # Summary
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"=== FINAL RESULTS ===", flush=True)
    for r in results:
        print(f"  {r['task_id']}: score={r['score']:.2f} success={r['success']}", flush=True)
    print(f"  average_score={avg_score:.2f}", flush=True)


if __name__ == "__main__":
    main()
