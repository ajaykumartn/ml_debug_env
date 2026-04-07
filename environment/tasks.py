"""
Task definitions for the ML Training Debugger environment.
"""
from typing import Dict, Any

TASKS: Dict[str, Dict[str, Any]] = {
    "easy_lr_divergence": {
        "id": "easy_lr_divergence",
        "difficulty": "easy",
        "title": "Fix Diverging Training Loss",
        "description": (
            "A model's training loss is diverging to infinity. "
            "Inspect the training logs and configuration, identify the root cause, "
            "and apply the correct fix to stabilize training."
        ),
        "bug": "high_learning_rate",
        "max_steps": 10,
        "correct_diagnosis": "learning_rate_too_high",
        "correct_fix": {"action": "modify_config", "key": "learning_rate", "value": 0.001},
        "grader_type": "easy",
    },
    "medium_wrong_loss": {
        "id": "medium_wrong_loss",
        "difficulty": "medium",
        "title": "Fix Wrong Loss Function for Classification",
        "description": (
            "A classification model's accuracy is stuck near the random baseline "
            "despite many epochs of training. The loss function is incompatible "
            "with the task type. Identify and fix it."
        ),
        "bug": "wrong_loss_function",
        "max_steps": 12,
        "correct_diagnosis": "wrong_loss_function",
        "correct_fix": {"action": "modify_config", "key": "loss_function", "value": "cross_entropy"},
        "grader_type": "medium",
    },    "medium_data_leakage": {
        "id": "medium_data_leakage",
        "difficulty": "medium",
        "title": "Detect and Fix Data Leakage",
        "description": (
            "A model shows suspiciously high validation accuracy and very low "
            "validation loss compared to training loss. Investigate the data "
            "pipeline, identify the leakage, and fix the data split."
        ),
        "bug": "data_leakage",
        "max_steps": 15,
        "correct_diagnosis": "data_leakage",
        "correct_fix": {
            "action": "apply_fix",
            "fix_type": "fix_data_split",
            "train": 0.8,
            "val": 0.2,
        },
        "grader_type": "medium",
    },
    "hard_overfitting_cascade": {
        "id": "hard_overfitting_cascade",
        "difficulty": "hard",
        "title": "Diagnose and Fix Cascading Overfitting",
        "description": (
            "A model is severely overfitting: training accuracy near 99% while "
            "validation accuracy drops below 40%. Multiple missing regularization "
            "settings are contributing. Identify all root causes and apply the "
            "complete sequence of fixes."
        ),
        "bug": "overfitting_cascade",
        "max_steps": 20,
        "correct_diagnosis": "overfitting_cascade",
        "correct_fix_sequence": [
            {"action": "modify_config", "key": "dropout_rate", "value": 0.3},
            {"action": "modify_config", "key": "weight_decay", "value": 0.001},
            {"action": "modify_config", "key": "epochs", "value": 20},
        ],
        "grader_type": "hard",
    },
    "hard_dual_bug": {
        "id": "hard_dual_bug",
        "difficulty": "hard",
        "title": "Debug Two Simultaneous Training Failures",
        "description": (
            "A model exhibits both overfitting AND periodic loss spikes from an "
            "unstable learning rate. Two bugs are active simultaneously with "
            "overlapping symptoms. Identify the primary root cause and apply "
            "the full sequence of fixes."
        ),
        "bug": "dual_bug_overfit_lr",
        "max_steps": 25,
        "correct_diagnosis": "overfitting_cascade",
        "correct_fix_sequence": [
            {"action": "modify_config", "key": "dropout_rate", "value": 0.3},
            {"action": "modify_config", "key": "weight_decay", "value": 0.001},
            {"action": "modify_config", "key": "epochs", "value": 20},
            {"action": "modify_config", "key": "learning_rate", "value": 0.001},
        ],
        "grader_type": "hard",
    },
}
