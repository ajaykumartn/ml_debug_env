"""
Task graders — deterministic scoring for each task.
All graders return a float in [0.0, 1.0].
"""
from typing import Dict, Any, List


def _values_close(a: Any, b: Any, tol: float = 0.05) -> bool:
    """Check if two values are equal or numerically close."""
    if str(a) == str(b):
        return True
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def grade_easy_task(
    correct_diagnosis: str,
    agent_diagnosis: str,
    fixes_applied: List[Dict[str, Any]],
    correct_fix_key: str,
    correct_fix_value: Any,
    steps_taken: int,
) -> Dict[str, Any]:
    """
    Easy task grader: single bug, single fix.
    - 0.4 for correct diagnosis
    - 0.4 for correct fix applied
    - 0.2 efficiency bonus (fewer steps = better)
    """
    score = 0.0
    breakdown = {}

    # Diagnosis score
    if agent_diagnosis == correct_diagnosis:
        breakdown["diagnosis"] = 0.4
    else:
        breakdown["diagnosis"] = 0.0
    score += breakdown["diagnosis"]

    # Fix score
    fix_correct = any(
        f.get("key") == correct_fix_key and _values_close(f.get("value"), correct_fix_value)
        for f in fixes_applied
    )
    breakdown["fix_applied"] = 0.4 if fix_correct else 0.0
    score += breakdown["fix_applied"]

    # Efficiency bonus
    if steps_taken <= 3:
        breakdown["efficiency"] = 0.2
    elif steps_taken <= 6:
        breakdown["efficiency"] = 0.1
    else:
        breakdown["efficiency"] = 0.0
    score += breakdown["efficiency"]

    return {
        "score": round(min(score, 1.0), 4),
        "breakdown": breakdown,
        "passed": score >= 0.8,
    }


def grade_medium_task(
    correct_diagnosis: str,
    agent_diagnosis: str,
    fixes_applied: List[Dict[str, Any]],
    correct_fix: Dict[str, Any],
    steps_taken: int,
    inspected_signals: List[str],
) -> Dict[str, Any]:
    """
    Medium task grader: requires reasoning across multiple signals.
    - 0.3 for correct diagnosis
    - 0.4 for correct fix
    - 0.2 for inspecting relevant signals (logs + metrics + data)
    - 0.1 efficiency bonus
    """
    score = 0.0
    breakdown = {}

    # Diagnosis
    breakdown["diagnosis"] = 0.3 if agent_diagnosis == correct_diagnosis else 0.0
    score += breakdown["diagnosis"]

    # Fix
    if correct_fix.get("action") == "modify_config":
        fix_correct = any(
            f.get("key") == correct_fix.get("key") and
            _values_close(f.get("value"), correct_fix.get("value"))
            for f in fixes_applied
        )
    elif correct_fix.get("action") == "apply_fix":
        fix_correct = any(
            f.get("fix_type") == correct_fix.get("fix_type")
            for f in fixes_applied
        )
    else:
        fix_correct = False
    breakdown["fix_applied"] = 0.4 if fix_correct else 0.0
    score += breakdown["fix_applied"]

    # Signal inspection (partial credit for investigating right areas)
    required_signals = {"logs", "metrics", "data"}
    inspected = set(inspected_signals)
    overlap = len(required_signals & inspected) / len(required_signals)
    breakdown["signal_inspection"] = round(0.2 * overlap, 4)
    score += breakdown["signal_inspection"]

    # Efficiency
    if steps_taken <= 5:
        breakdown["efficiency"] = 0.1
    elif steps_taken <= 8:
        breakdown["efficiency"] = 0.05
    else:
        breakdown["efficiency"] = 0.0
    score += breakdown["efficiency"]

    return {
        "score": round(min(score, 1.0), 4),
        "breakdown": breakdown,
        "passed": score >= 0.7,
    }


def grade_hard_task(
    correct_diagnosis: str,
    agent_diagnosis: str,
    fixes_applied: List[Dict[str, Any]],
    correct_fix_sequence: List[Dict[str, Any]],
    steps_taken: int,
    inspected_signals: List[str],
) -> Dict[str, Any]:
    """
    Hard task grader: multiple interacting bugs, sequence of fixes required.
    - 0.25 for correct diagnosis
    - 0.45 for fix sequence (partial credit per correct fix)
    - 0.2 for thorough signal inspection
    - 0.1 efficiency bonus
    """
    score = 0.0
    breakdown = {}

    # Diagnosis
    breakdown["diagnosis"] = 0.25 if agent_diagnosis == correct_diagnosis else 0.0
    score += breakdown["diagnosis"]

    # Fix sequence — partial credit per fix
    per_fix_score = 0.45 / len(correct_fix_sequence)
    fix_score = 0.0
    for correct_fix in correct_fix_sequence:
        if correct_fix.get("action") == "modify_config":
            matched = any(
                f.get("key") == correct_fix.get("key") and
                _values_close(f.get("value"), correct_fix.get("value"))
                for f in fixes_applied
            )
        elif correct_fix.get("action") == "apply_fix":
            matched = any(
                f.get("fix_type") == correct_fix.get("fix_type")
                for f in fixes_applied
            )
        else:
            matched = False
        if matched:
            fix_score += per_fix_score
    breakdown["fix_sequence"] = round(fix_score, 4)
    score += breakdown["fix_sequence"]

    # Signal inspection
    required_signals = {"logs", "metrics", "config", "data"}
    inspected = set(inspected_signals)
    overlap = len(required_signals & inspected) / len(required_signals)
    breakdown["signal_inspection"] = round(0.2 * overlap, 4)
    score += breakdown["signal_inspection"]

    # Efficiency
    if steps_taken <= 8:
        breakdown["efficiency"] = 0.1
    elif steps_taken <= 12:
        breakdown["efficiency"] = 0.05
    else:
        breakdown["efficiency"] = 0.0
    score += breakdown["efficiency"]

    return {
        "score": round(min(score, 1.0), 4),
        "breakdown": breakdown,
        "passed": score >= 0.6,
    }
