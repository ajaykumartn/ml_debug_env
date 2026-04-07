"""
MLDebugEnv — OpenEnv-compliant ML Training Debugger Environment
"""
from typing import Any, Dict, List, Optional, Tuple
import copy

from .models import Observation, Action, Reward, TrainingConfig
from .simulator import generate_scenario
from .tasks import TASKS
from .graders import grade_easy_task, grade_medium_task, grade_hard_task


AVAILABLE_ACTIONS = [
    "inspect_logs",
    "inspect_metrics",
    "inspect_config",
    "check_data",
    "diagnose_issue",
    "modify_config",
    "apply_fix",
    "restart_training",
    "submit_diagnosis",
]

VALID_DIAGNOSES = [
    "learning_rate_too_high",
    "learning_rate_too_low",
    "wrong_loss_function",
    "data_leakage",
    "vanishing_gradient",
    "exploding_gradient_optimizer_mismatch",
    "overfitting_cascade",
    "underfitting",
    "batch_size_issue",
    "scheduler_misconfiguration",
]


class MLDebugEnv:
    """
    OpenEnv environment simulating broken ML training runs.
    The agent must diagnose and fix issues to restore healthy training.
    """

    metadata = {
        "name": "ml-training-debugger",
        "version": "1.0.0",
        "description": "Debug broken ML training runs by inspecting logs, metrics, and configs",
        "tasks": list(TASKS.keys()),
    }

    def __init__(self, task_id: Optional[str] = None):
        self._task_id = task_id or "easy_lr_divergence"
        self._task = TASKS[self._task_id]
        self._scenario: Optional[Dict[str, Any]] = None
        self._step_count = 0
        self._done = False
        self._initialized = False
        self._seed: Optional[int] = None
        self._diagnosis: Optional[str] = None
        self._fixes_applied: List[Dict[str, Any]] = []
        self._inspected_signals: List[str] = []
        self._diagnosis_history: List[str] = []
        self._fix_history: List[str] = []
        self._current_config: Optional[TrainingConfig] = None
        self._cumulative_reward = 0.0
        self._last_action_error: Optional[str] = None

    # ── OpenEnv Interface ──────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset environment to initial broken state. Pass seed for reproducibility."""
        self._seed = seed
        self._scenario = generate_scenario(self._task["bug"], seed=seed)
        self._step_count = 0
        self._done = False
        self._diagnosis = None
        self._fixes_applied = []
        self._inspected_signals = []
        self._diagnosis_history = []
        self._fix_history = []
        self._current_config = copy.deepcopy(self._scenario["config"])
        self._cumulative_reward = 0.0
        self._last_action_error = None
        self._initialized = True
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Execute one agent action and return (observation, reward, done, info)."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step_count += 1
        self._last_action_error = None
        reward_value, reward_breakdown, reward_msg = self._execute_action(action)

        # Check termination
        max_steps = self._task["max_steps"]
        if self._step_count >= max_steps:
            self._done = True
            reward_msg += f" | Episode ended: max steps ({max_steps}) reached."

        # Clamp reward to [0.0, 1.0] for storage; negative values penalize cumulative
        clamped = round(max(0.0, min(reward_value, 1.0)), 4)
        reward = Reward(
            value=clamped,
            breakdown=reward_breakdown,
            message=reward_msg,
        )
        self._cumulative_reward += reward_value  # track raw (including negatives)

        obs = self._build_observation()
        info = {
            "step": self._step_count,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "last_action_error": self._last_action_error,
            "task_id": self._task_id,
        }
        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return full internal state (for debugging/evaluation)."""
        return {
            "task_id": self._task_id,
            "step": self._step_count,
            "done": self._done,
            "seed": self._seed,
            "diagnosis": self._diagnosis,
            "fixes_applied": self._fixes_applied,
            "inspected_signals": self._inspected_signals,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "current_config": self._current_config.model_dump() if self._current_config else None,
        }

    def close(self) -> Dict[str, Any]:
        """Finalize episode and return graded score."""
        if not self._initialized:
            return {"score": 0.0, "breakdown": {}, "passed": False}
        return self._compute_final_score()

    # ── Action Execution ───────────────────────────────────────────────────────

    def _execute_action(self, action: Action) -> Tuple[float, Dict[str, float], str]:
        atype = action.action_type
        params = action.parameters

        if atype == "inspect_logs":
            self._inspected_signals.append("logs")
            return 0.05, {"exploration": 0.05}, "Inspected training logs."

        elif atype == "inspect_metrics":
            self._inspected_signals.append("metrics")
            return 0.05, {"exploration": 0.05}, "Inspected training metrics."

        elif atype == "inspect_config":
            self._inspected_signals.append("config")
            return 0.05, {"exploration": 0.05}, "Inspected training configuration."

        elif atype == "check_data":
            self._inspected_signals.append("data")
            return 0.05, {"exploration": 0.05}, "Checked data pipeline and splits."

        elif atype == "diagnose_issue":
            diagnosis = params.get("diagnosis", "")
            if diagnosis not in VALID_DIAGNOSES:
                self._last_action_error = f"Unknown diagnosis: {diagnosis}"
                return 0.0, {"diagnosis": 0.0}, f"Invalid diagnosis: {diagnosis}"

            self._diagnosis = diagnosis
            self._diagnosis_history.append(diagnosis)
            correct = self._task["correct_diagnosis"]

            if diagnosis == correct:
                return 0.2, {"diagnosis": 0.2}, f"Correct diagnosis: {diagnosis}"
            else:
                return -0.05, {"diagnosis": -0.05}, f"Incorrect diagnosis: {diagnosis}. Keep investigating."

        elif atype == "modify_config":
            key = params.get("key")
            value = params.get("value")
            if not key or value is None:
                self._last_action_error = "modify_config requires 'key' and 'value'"
                return 0.0, {"fix": 0.0}, "Invalid modify_config parameters."

            # Apply to current config
            try:
                config_dict = self._current_config.model_dump()
                config_dict[key] = value
                self._current_config = TrainingConfig(**config_dict)
                self._fixes_applied.append({"action": "modify_config", "key": key, "value": value})
                self._fix_history.append(f"modify_config: {key}={value}")

                # Partial reward if fix is in the right direction
                reward = self._evaluate_fix_reward({"action": "modify_config", "key": key, "value": value})
                return reward, {"fix": reward}, f"Applied config change: {key}={value}"
            except Exception as e:
                self._last_action_error = str(e)
                return 0.0, {"fix": 0.0}, f"Config change failed: {e}"

        elif atype == "apply_fix":
            fix_type = params.get("fix_type")
            if not fix_type:
                self._last_action_error = "apply_fix requires 'fix_type'"
                return 0.0, {"fix": 0.0}, "Invalid apply_fix parameters."

            self._fixes_applied.append({"action": "apply_fix", "fix_type": fix_type, **params})
            self._fix_history.append(f"apply_fix: {fix_type}")
            reward = self._evaluate_fix_reward({"action": "apply_fix", "fix_type": fix_type})
            return reward, {"fix": reward}, f"Applied fix: {fix_type}"

        elif atype == "restart_training":
            # Neutral action — no reward, no penalty
            return 0.0, {"restart": 0.0}, "Training restarted with current configuration."

        elif atype == "submit_diagnosis":
            # Final submission — triggers grading
            self._done = True
            result = self._compute_final_score()
            score = result["score"]
            return score, {"final_score": score}, f"Diagnosis submitted. Final score: {score}"

        else:
            self._last_action_error = f"Unknown action type: {atype}"
            return -0.1, {"invalid": -0.1}, f"Unknown action: {atype}. Penalty applied."

    def _evaluate_fix_reward(self, fix: Dict[str, Any]) -> float:
        """Give partial reward if fix matches expected correct fix."""
        grader_type = self._task["grader_type"]

        def _close(a: Any, b: Any) -> bool:
            if str(a) == str(b):
                return True
            try:
                return abs(float(a) - float(b)) <= 0.05
            except (TypeError, ValueError):
                return False

        if grader_type == "easy":
            correct = self._task["correct_fix"]
            if fix.get("key") == correct.get("key") and _close(fix.get("value"), correct.get("value")):
                return 0.3
            return 0.0

        elif grader_type == "medium":
            correct = self._task["correct_fix"]
            if fix.get("action") == "apply_fix" and fix.get("fix_type") == correct.get("fix_type"):
                return 0.3
            if fix.get("key") == correct.get("key") and _close(fix.get("value"), correct.get("value")):
                return 0.3
            return 0.0

        elif grader_type == "hard":
            sequence = self._task["correct_fix_sequence"]
            for correct in sequence:
                if fix.get("key") == correct.get("key") and _close(fix.get("value"), correct.get("value")):
                    return 0.15
            return 0.0

        return 0.0

    def _compute_final_score(self) -> Dict[str, Any]:
        """Run the appropriate grader and return final score."""
        grader_type = self._task["grader_type"]
        diagnosis = self._diagnosis or ""

        if grader_type == "easy":
            correct_fix = self._task["correct_fix"]
            result = grade_easy_task(
                correct_diagnosis=self._task["correct_diagnosis"],
                agent_diagnosis=diagnosis,
                fixes_applied=self._fixes_applied,
                correct_fix_key=correct_fix["key"],
                correct_fix_value=correct_fix["value"],
                steps_taken=self._step_count,
            )

        elif grader_type == "medium":
            result = grade_medium_task(
                correct_diagnosis=self._task["correct_diagnosis"],
                agent_diagnosis=diagnosis,
                fixes_applied=self._fixes_applied,
                correct_fix=self._task["correct_fix"],
                steps_taken=self._step_count,
                inspected_signals=list(set(self._inspected_signals)),
            )

        elif grader_type == "hard":
            result = grade_hard_task(
                correct_diagnosis=self._task["correct_diagnosis"],
                agent_diagnosis=diagnosis,
                fixes_applied=self._fixes_applied,
                correct_fix_sequence=self._task["correct_fix_sequence"],
                steps_taken=self._step_count,
                inspected_signals=list(set(self._inspected_signals)),
            )
        else:
            result = {"score": 0.0, "breakdown": {}, "passed": False}

        return result

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        scenario = self._scenario
        metrics = scenario["metrics"]
        # Reveal metrics progressively as steps increase
        visible = metrics[:min(self._step_count + 3, len(metrics))]
        return Observation(
            task_id=self._task_id,
            step=self._step_count,
            training_logs=scenario["logs"],
            metrics_history=visible,
            current_config=self._current_config,
            system_alerts=scenario["alerts"],
            available_actions=AVAILABLE_ACTIONS,
            diagnosis_history=list(self._diagnosis_history),
            fix_history=list(self._fix_history),
            is_training_healthy=self._is_healthy(),
        )

    def _is_healthy(self) -> bool:
        """Check if applied fixes resolve the issue."""
        if not self._initialized:
            return False
        result = self._compute_final_score()
        return result["score"] >= 0.8
