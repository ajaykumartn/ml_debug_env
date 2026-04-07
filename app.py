"""
FastAPI server exposing the MLDebugEnv via OpenEnv-compliant HTTP endpoints.
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn

from environment import MLDebugEnv
from environment.models import Action
from environment.tasks import TASKS

app = FastAPI(
    title="ML Training Debugger — OpenEnv",
    description="Debug broken ML training runs. OpenEnv-compliant environment.",
    version="1.0.0",
)

# One env instance per task (keyed by task_id)
_envs: Dict[str, MLDebugEnv] = {}


def _get_env(task_id: str) -> MLDebugEnv:
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    if task_id not in _envs:
        _envs[task_id] = MLDebugEnv(task_id=task_id)
    return _envs[task_id]


class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy_lr_divergence"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    task_id: Optional[str] = "easy_lr_divergence"
    action: Action


class CloseRequest(BaseModel):
    task_id: Optional[str] = "easy_lr_divergence"


@app.get("/")
def root():
    return {
        "name": "ml-training-debugger",
        "version": "1.0.0",
        "tasks": list(TASKS.keys()),
        "endpoints": ["/reset", "/step", "/state", "/close", "/health"],
    }


@app.get("/health")
def health():
    """Liveness probe — automated validator pings this."""
    return {"status": "ok", "env": "ml-training-debugger"}


@app.get("/reset")
def reset_get(task_id: str = Query(default="easy_lr_divergence"), seed: Optional[int] = Query(default=None)):
    """GET reset — for automated validator ping compatibility."""
    env = _get_env(task_id)
    obs = env.reset(seed=seed)
    return obs.model_dump()


@app.post("/reset")
def reset_post(req: ResetRequest):
    env = _get_env(req.task_id)
    obs = env.reset(seed=req.seed)
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    env = _get_env(req.task_id)
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(task_id: str = Query(default="easy_lr_divergence")):
    env = _get_env(task_id)
    return env.state()


@app.post("/close")
def close(req: CloseRequest):
    env = _get_env(req.task_id)
    result = env.close()
    return result


@app.get("/validate")
def validate():
    """OpenEnv validation endpoint — confirms spec compliance."""
    return {
        "name": "ml-training-debugger",
        "version": "1.0.0",
        "spec": "openenv",
        "tasks": [
            {"id": tid, "difficulty": t["difficulty"]}
            for tid, t in TASKS.items()
        ],
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "close": "POST /close",
        },
        "reward_range": [0.0, 1.0],
        "status": "valid",
    }


@app.get("/tasks")
def list_tasks():
    return {
        tid: {
            "difficulty": t["difficulty"],
            "title": t["title"],
            "description": t["description"],
        }
        for tid, t in TASKS.items()
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
