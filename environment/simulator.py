"""
ML Training Run Simulator
Generates realistic broken training scenarios with logs, metrics, and configs.
Supports seeding for reproducibility + randomized variations per reset.
"""
import random
from typing import Dict, List, Any, Optional
from .models import TrainingConfig, TrainingMetrics


BUGS = {
    "high_learning_rate": {
        "description": "Learning rate is too high causing loss divergence",
        "difficulty": "easy",
        "correct_diagnosis": "learning_rate_too_high",
    },
    "wrong_loss_function": {
        "description": "Using MSE loss for multi-class classification — accuracy stuck near random",
        "difficulty": "medium",
        "correct_diagnosis": "wrong_loss_function",
    },
    "data_leakage": {
        "description": "Validation data leaking into training — val metrics unrealistically good",
        "difficulty": "medium",
        "correct_diagnosis": "data_leakage",
    },
    "overfitting_cascade": {
        "description": "Severe overfitting: no dropout + no weight decay + too many epochs",
        "difficulty": "hard",
        "correct_diagnosis": "overfitting_cascade",
    },
    "dual_bug_overfit_lr": {
        "description": "Two simultaneous bugs: overfitting AND learning rate instability",
        "difficulty": "hard",
        "correct_diagnosis": "overfitting_cascade",
    },
}

# ── Log template pools (randomized per reset) ──────────────────────────────────

_LR_LOGS = [
    [
        "Epoch 1/{e} — train_loss={l1:.4f} — WARN: loss increasing",
        "Epoch 2/{e} — train_loss={l2:.4f} — WARN: loss diverging",
        "Epoch 3/{e} — train_loss={l3:.4f} — ERROR: loss is NaN or Inf detected",
        "Epoch 4/{e} — train_loss=inf — CRITICAL: training unstable",
        "Gradient norm: {gn:.1f} — WARN: gradients exploding",
    ],
    [
        "Training started with lr={lr}",
        "Epoch 1/{e} — loss={l1:.4f} — gradient_norm={gn:.1f}",
        "Epoch 2/{e} — loss={l2:.4f} — WARN: loss not decreasing",
        "Epoch 3/{e} — loss={l3:.4f} — ERROR: NaN detected in loss",
        "CRITICAL: Training diverged at epoch 3 — check learning rate",
    ],
    [
        "Initializing optimizer: SGD lr={lr}",
        "Epoch 1/{e} — train_loss={l1:.4f} val_loss={l1v:.4f}",
        "Epoch 2/{e} — train_loss={l2:.4f} — WARN: loss exploding",
        "CRITICAL: Loss exceeded 1e6 — training halted",
        "Gradient norm at halt: {gn:.1f} — exceeds safe threshold of 10.0",
    ],
]

_OVERFIT_LOGS = [
    [
        "Epoch 10/{e} — train_loss={tl:.4f} val_loss={vl:.4f}",
        "Epoch 20/{e} — train_loss={tl2:.4f} val_loss={vl2:.4f}",
        "Epoch 30/{e} — train_loss={tl3:.4f} val_loss={vl3:.4f}",
        "CRITICAL: val_loss increasing while train_loss decreasing",
        "WARN: train_acc={ta:.2f} val_acc={va:.2f} — severe overfitting",
        "INFO: dropout_rate=0.0, weight_decay=0.0",
    ],
    [
        "Model is memorizing training data",
        "Epoch 15/{e} — train_acc={ta:.2f} val_acc={va:.2f}",
        "Epoch 25/{e} — train_loss={tl:.4f} val_loss={vl:.4f} — gap widening",
        "CRITICAL: Generalization gap = {gap:.2f} — overfitting confirmed",
        "WARN: No regularization detected (dropout=0.0, weight_decay=0.0)",
        "INFO: epochs={e} may be excessive for this dataset size",
    ],
]

_LEAKAGE_LOGS = [
    [
        "Data split: train={tr}%, val={va}%",
        "WARNING: Detected {ov}% overlap between train and val indices",
        "Epoch 3/10 — train_loss={tl:.4f} val_loss={vl:.4f} — SUSPICIOUS",
        "Epoch 5/10 — val_acc={vacc:.4f} — WARN: val accuracy unrealistically high",
        "Epoch 10/10 — val_loss={vl2:.4f} — CRITICAL: val_loss << train_loss",
    ],
    [
        "Dataset loader initialized",
        "WARN: train/val split overlap detected in index sampler",
        "Epoch 2/10 — val_loss={vl:.4f} train_loss={tl:.4f} — ratio suspicious",
        "Epoch 5/10 — val_acc={vacc:.4f} — exceeds expected ceiling for this task",
        "CRITICAL: Validation metrics are not trustworthy — possible data contamination",
    ],
]


# ── Metric generators ──────────────────────────────────────────────────────────

def _diverging_metrics(epochs: int, lr: float) -> List[TrainingMetrics]:
    loss = 2.5
    metrics = []
    for e in range(1, epochs + 1):
        loss *= (1 + lr * random.uniform(0.3, 0.7))
        metrics.append(TrainingMetrics(
            epoch=e,
            train_loss=round(min(loss, 1e6), 4),
            val_loss=round(min(loss * random.uniform(0.95, 1.05), 1e6), 4),
            gradient_norm=round(random.uniform(80, 300), 2),
            learning_rate_actual=lr,
        ))
    return metrics


def _wrong_loss_metrics(epochs: int) -> List[TrainingMetrics]:
    loss = 0.25
    metrics = []
    for e in range(1, epochs + 1):
        loss -= random.uniform(0.001, 0.003)
        metrics.append(TrainingMetrics(
            epoch=e,
            train_loss=round(max(loss, 0.20), 4),
            val_loss=round(max(loss + random.uniform(0.01, 0.03), 0.22), 4),
            train_accuracy=round(random.uniform(0.29, 0.36), 4),
            val_accuracy=round(random.uniform(0.27, 0.34), 4),
            learning_rate_actual=0.001,
        ))
    return metrics


def _leakage_metrics(epochs: int) -> List[TrainingMetrics]:
    train_loss = 1.8
    metrics = []
    for e in range(1, epochs + 1):
        train_loss -= random.uniform(0.05, 0.12)
        val_loss = train_loss * random.uniform(0.25, 0.45)
        metrics.append(TrainingMetrics(
            epoch=e,
            train_loss=round(max(train_loss, 0.3), 4),
            val_loss=round(max(val_loss, 0.04), 4),
            train_accuracy=round(0.5 + e * 0.02, 4),
            val_accuracy=round(min(0.85 + random.uniform(0, 0.12), 0.99), 4),
            learning_rate_actual=0.001,
        ))
    return metrics


def _overfitting_metrics(epochs: int) -> List[TrainingMetrics]:
    train_loss, val_loss = 2.0, 2.1
    metrics = []
    for e in range(1, epochs + 1):
        train_loss -= random.uniform(0.05, 0.1)
        val_loss += random.uniform(0.02, 0.07)
        metrics.append(TrainingMetrics(
            epoch=e,
            train_loss=round(max(train_loss, 0.01), 4),
            val_loss=round(val_loss, 4),
            train_accuracy=round(min(0.5 + e * 0.025, 0.99), 4),
            val_accuracy=round(max(0.65 - e * 0.012, 0.28), 4),
            learning_rate_actual=0.001,
        ))
    return metrics


def _dual_bug_metrics(epochs: int) -> List[TrainingMetrics]:
    """Overfitting + occasional LR spikes — two bugs active."""
    train_loss, val_loss = 2.0, 2.1
    metrics = []
    for e in range(1, epochs + 1):
        train_loss -= random.uniform(0.04, 0.09)
        val_loss += random.uniform(0.02, 0.06)
        # Occasional spike from LR instability
        spike = random.uniform(0.3, 0.8) if e % 5 == 0 else 0.0
        metrics.append(TrainingMetrics(
            epoch=e,
            train_loss=round(max(train_loss + spike, 0.01), 4),
            val_loss=round(val_loss, 4),
            train_accuracy=round(min(0.5 + e * 0.022, 0.98), 4),
            val_accuracy=round(max(0.63 - e * 0.011, 0.25), 4),
            gradient_norm=round(random.uniform(5, 150) if e % 5 == 0 else random.uniform(1, 10), 2),
            learning_rate_actual=0.001,
        ))
    return metrics


# ── Main scenario generator ────────────────────────────────────────────────────

def generate_scenario(bug_name: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate a full broken training scenario.
    If seed is provided, the scenario is fully reproducible.
    """
    if seed is not None:
        random.seed(seed)

    bug = BUGS[bug_name]

    if bug_name == "high_learning_rate":
        lr = round(random.choice([0.3, 0.5, 0.8, 1.0]), 1)
        epochs = random.choice([8, 10, 12])
        config = TrainingConfig(
            learning_rate=lr,
            batch_size=random.choice([16, 32]),
            optimizer=random.choice(["sgd", "sgd"]),
            loss_function="cross_entropy",
            epochs=epochs,
            dropout_rate=0.2,
            weight_decay=0.0,
        )
        metrics = _diverging_metrics(epochs, lr)
        l1 = metrics[0].train_loss
        l2 = metrics[1].train_loss if len(metrics) > 1 else l1 * 1.5
        l3 = metrics[2].train_loss if len(metrics) > 2 else l2 * 1.5
        gn = metrics[0].gradient_norm or 187.3
        template = random.choice(_LR_LOGS)
        logs = [
            line.format(e=epochs, lr=lr, l1=l1, l2=l2, l3=l3,
                        l1v=l1 * 0.98, gn=gn)
            for line in template
        ]
        alerts = [
            "CRITICAL: Training loss is diverging (NaN/Inf detected)",
            f"WARN: Gradient norm > 100 — possible exploding gradients",
            f"WARN: Current learning_rate={lr} may be too high",
        ]

    elif bug_name == "wrong_loss_function":
        epochs = random.choice([12, 15, 18])
        n_classes = random.choice([5, 8, 10])
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=random.choice([32, 64]),
            optimizer="adam",
            loss_function="mse",
            epochs=epochs,
            dropout_rate=0.2,
            weight_decay=0.0,
        )
        metrics = _wrong_loss_metrics(epochs)
        baseline_acc = round(1.0 / n_classes, 2)
        logs = [
            f"Task type: multi-class classification ({n_classes} classes)",
            "Loss function: MSELoss — applied to raw class logits",
            f"Epoch 5/{epochs} — train_acc={metrics[4].train_accuracy} val_acc={metrics[4].val_accuracy}",
            f"Epoch {epochs//2}/{epochs} — train_acc={metrics[epochs//2 - 1].train_accuracy}",
            f"WARN: Accuracy not improving after {epochs//2} epochs",
            "WARN: Loss plateau detected — model not learning effectively",
        ]
        alerts = [
            f"WARN: Accuracy stuck near random baseline ({baseline_acc} for {n_classes}-class problem)",
            "WARN: Loss function may be incompatible with task type",
            f"INFO: Task is multi-class classification ({n_classes} classes)",
            "INFO: Consider using CrossEntropyLoss for classification tasks",
        ]

    elif bug_name == "data_leakage":
        overlap = random.choice([25, 30, 35])
        train_pct = 60
        val_pct = 40
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=32,
            optimizer="adam",
            loss_function="cross_entropy",
            epochs=10,
            dropout_rate=0.2,
            weight_decay=0.0,
            data_split={"train": train_pct / 100, "val": val_pct / 100, "overlap": overlap / 100},
        )
        metrics = _leakage_metrics(10)
        template = random.choice(_LEAKAGE_LOGS)
        logs = [
            line.format(
                tr=train_pct, va=val_pct, ov=overlap,
                tl=metrics[2].train_loss, vl=metrics[2].val_loss,
                vacc=metrics[4].val_accuracy, vl2=metrics[9].val_loss,
            )
            for line in template
        ]
        alerts = [
            "CRITICAL: Validation loss is significantly lower than training loss",
            f"WARN: Val accuracy ({metrics[4].val_accuracy}) is unrealistically high",
            f"WARN: {overlap}% overlap detected between train and val indices",
            "WARN: Model evaluation metrics are not trustworthy",
        ]

    elif bug_name == "overfitting_cascade":
        epochs = random.choice([80, 100, 120])
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=32,
            optimizer="adam",
            loss_function="cross_entropy",
            epochs=epochs,
            dropout_rate=0.0,
            weight_decay=0.0,
        )
        metrics = _overfitting_metrics(30)
        m10 = metrics[9]
        m20 = metrics[19]
        m30 = metrics[29]
        gap = round(m30.val_loss - m30.train_loss, 4)
        template = random.choice(_OVERFIT_LOGS)
        logs = [
            line.format(
                e=epochs,
                tl=m10.train_loss, vl=m10.val_loss,
                tl2=m20.train_loss, vl2=m20.val_loss,
                tl3=m30.train_loss, vl3=m30.val_loss,
                ta=m30.train_accuracy, va=m30.val_accuracy,
                gap=gap,
            )
            for line in template
        ]
        alerts = [
            f"CRITICAL: Severe overfitting — train_acc={m30.train_accuracy} val_acc={m30.val_accuracy}",
            "WARN: No dropout configured (dropout_rate=0.0)",
            "WARN: No weight decay / L2 regularization (weight_decay=0.0)",
            f"WARN: Epoch count ({epochs}) may be excessive for dataset size",
        ]

    elif bug_name == "dual_bug_overfit_lr":
        epochs = random.choice([80, 100])
        lr = round(random.choice([0.05, 0.08, 0.1]), 2)
        config = TrainingConfig(
            learning_rate=lr,
            batch_size=32,
            optimizer="adam",
            loss_function="cross_entropy",
            epochs=epochs,
            dropout_rate=0.0,
            weight_decay=0.0,
        )
        metrics = _dual_bug_metrics(25)
        m10 = metrics[9]
        m20 = metrics[19]
        m25 = metrics[24]
        logs = [
            f"Training config: lr={lr}, epochs={epochs}, dropout=0.0, weight_decay=0.0",
            f"Epoch 5/{epochs} — train_loss={metrics[4].train_loss} val_loss={metrics[4].val_loss}",
            f"Epoch 10/{epochs} — train_loss={m10.train_loss} val_loss={m10.val_loss} grad_norm={m10.gradient_norm}",
            f"Epoch 15/{epochs} — WARN: loss spike detected (gradient_norm={metrics[14].gradient_norm})",
            f"Epoch 20/{epochs} — train_acc={m20.train_accuracy} val_acc={m20.val_accuracy}",
            f"Epoch 25/{epochs} — train_acc={m25.train_accuracy} val_acc={m25.val_accuracy}",
            "CRITICAL: val_loss diverging from train_loss — overfitting confirmed",
            f"WARN: Periodic loss spikes suggest lr={lr} may be unstable",
            "WARN: dropout_rate=0.0, weight_decay=0.0 — no regularization",
        ]
        alerts = [
            f"CRITICAL: Overfitting detected — train_acc={m25.train_accuracy} val_acc={m25.val_accuracy}",
            f"WARN: Periodic gradient spikes (lr={lr} may contribute)",
            "WARN: No regularization (dropout=0.0, weight_decay=0.0)",
            f"WARN: epochs={epochs} is excessive — early stopping recommended",
        ]

    else:
        raise ValueError(f"Unknown bug: {bug_name}")

    return {
        "bug_name": bug_name,
        "bug_info": bug,
        "config": config,
        "metrics": metrics,
        "logs": logs,
        "alerts": alerts,
    }
