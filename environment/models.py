from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class TrainingConfig(BaseModel):
    learning_rate: float
    batch_size: int
    optimizer: str
    loss_function: str
    epochs: int
    dropout_rate: float
    weight_decay: float
    scheduler: Optional[str] = None
    gradient_clip: Optional[float] = None
    data_split: Optional[Dict[str, float]] = None


class TrainingMetrics(BaseModel):
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    gradient_norm: Optional[float] = None
    learning_rate_actual: Optional[float] = None


class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    step: int
    training_logs: List[str] = Field(description="Recent training log lines")
    metrics_history: List[TrainingMetrics] = Field(description="Per-epoch metrics")
    current_config: TrainingConfig = Field(description="Current training configuration")
    system_alerts: List[str] = Field(description="Warnings and error alerts")
    available_actions: List[str] = Field(description="Actions the agent can take")
    diagnosis_history: List[str] = Field(description="Agent's previous diagnoses")
    fix_history: List[str] = Field(description="Agent's previous fixes applied")
    is_training_healthy: bool = Field(description="Whether training is currently healthy")


class Action(BaseModel):
    """What the agent can do."""
    action_type: str = Field(
        description=(
            "One of: inspect_logs, inspect_metrics, diagnose_issue, "
            "modify_config, restart_training, apply_fix, check_data, submit_diagnosis"
        )
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters"
    )


class Reward(BaseModel):
    """Reward signal returned after each step."""
    value: float = Field(description="Step reward, clamped to [0.0, 1.0] before storage")
    breakdown: Dict[str, float] = Field(description="Per-component reward breakdown")
    message: str = Field(description="Human-readable reward explanation")
