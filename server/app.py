"""FastAPI server — OpenEnv SQL Reviewer"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from sql_reviewer_env import SQLReviewerEnv, Action, TASKS

app = FastAPI(title="SQL Reviewer OpenEnv", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_sessions: dict[str, SQLReviewerEnv] = {}

def _env(task_id: str) -> SQLReviewerEnv:
    if task_id not in _sessions:
        _sessions[task_id] = SQLReviewerEnv(task_id)
    return _sessions[task_id]

@app.get("/")
def root():
    return {"name": "sql-reviewer-env", "tasks": list(TASKS), "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(task_id: str = "syntax_fix"):
    if task_id not in TASKS:
        raise HTTPException(400, f"Unknown task: {task_id}")
    env = SQLReviewerEnv(task_id)
    _sessions[task_id] = env
    return env.reset().model_dump()

@app.post("/step")
def step(action: Action, task_id: str = "syntax_fix"):
    env = _env(task_id)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": done, "info": info}

@app.get("/state")
def state(task_id: str = "syntax_fix"):
    return _env(task_id).state()

@app.get("/tasks")
def list_tasks():
    return {tid: {"difficulty": t["difficulty"], "description": t["description"]} for tid, t in TASKS.items()}

def main():
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 7860)))

if __name__ == "__main__":
    main()
