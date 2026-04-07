"""
OpenEnv-OpsFlow: FastAPI Application

Exposes the OpsFlowEnv through HTTP endpoints for the OpenEnv spec.
Endpoints: /reset, /step, /state, /tasks, /health
"""

import os
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import OpsFlowEnv
from models import Action, Observation, EnvironmentState, StepResult


# =============================================================================
# Request/Response Models
# =============================================================================

class ResetRequest(BaseModel):
    """Request body for /reset endpoint."""
    task_id: Optional[str] = None


class ResetResponse(BaseModel):
    """Response body for /reset endpoint."""
    observation: Observation
    task_id: str
    message: str


class StepRequest(BaseModel):
    """Request body for /step endpoint."""
    action: Action


class StepResponse(BaseModel):
    """Response body for /step endpoint."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResponse(BaseModel):
    """Response body for /state endpoint."""
    state: EnvironmentState


class TaskInfo(BaseModel):
    """Information about a task."""
    task_id: str
    difficulty: str
    description: str
    expected_steps: int
    requires_approval: bool


class TasksResponse(BaseModel):
    """Response body for /tasks endpoint."""
    tasks: list[TaskInfo]


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str
    environment: str
    version: str


# =============================================================================
# Application Setup
# =============================================================================

# Global environment instance
env: Optional[OpsFlowEnv] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global env
    # Startup
    env = OpsFlowEnv()
    yield
    # Shutdown
    env = None


app = FastAPI(
    title="OpenEnv-OpsFlow",
    description="Policy-Aware Customer Support & Compliance Orchestrator Environment",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for broader compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Check Endpoint
# =============================================================================

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        environment="OpsFlowEnv",
        version="1.0.0"
    )


# =============================================================================
# OpenEnv Core Endpoints
# =============================================================================

@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = ResetRequest()) -> ResetResponse:
    """
    Reset the environment to a fresh state.
    
    Optionally specify a task_id to load a specific task.
    If no task_id is provided, loads the first available task.
    """
    global env
    if env is None:
        env = OpsFlowEnv()
    
    try:
        observation = env.reset(task_id=request.task_id)
        task_id = env.current_task.task_id if env.current_task else "unknown"
        
        return ResetResponse(
            observation=observation,
            task_id=task_id,
            message=f"Environment reset with task: {task_id}"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest) -> StepResponse:
    """
    Execute an action in the environment.
    
    Returns the new observation, reward, done flag, and info dict.
    """
    global env
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    try:
        result = env.step(request.action)
        
        return StepResponse(
            observation=result.observation,
            reward=result.reward,
            done=result.done,
            info=result.info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state", response_model=StateResponse)
async def get_state() -> StateResponse:
    """
    Get the current complete environment state.
    
    Returns all internal state information.
    """
    global env
    if env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )
    
    try:
        state = env.state()
        return StateResponse(state=state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State retrieval failed: {str(e)}")


# =============================================================================
# Task Information Endpoints
# =============================================================================

@app.get("/tasks", response_model=TasksResponse)
async def list_tasks() -> TasksResponse:
    """
    List all available tasks.
    
    Returns task IDs, difficulties, and descriptions.
    """
    global env
    if env is None:
        env = OpsFlowEnv()
    
    task_ids = env.get_available_tasks()
    tasks = [
        TaskInfo(**env.get_task_info(task_id))
        for task_id in task_ids
    ]
    
    return TasksResponse(tasks=tasks)


@app.get("/tasks/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str) -> TaskInfo:
    """
    Get information about a specific task.
    """
    global env
    if env is None:
        env = OpsFlowEnv()
    
    try:
        info = env.get_task_info(task_id)
        return TaskInfo(**info)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)
