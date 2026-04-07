#!/usr/bin/env python3
"""
OpenEnv-OpsFlow: Baseline Inference Script

Uses OpenAI API client to run an LLM agent against all tasks.
Produces reproducible baseline scores with structured logging.

Required environment variables:
- OPENAI_API_KEY or API_KEY: API key for the LLM
- API_BASE_URL: Base URL for the API endpoint
- MODEL_NAME: Model identifier to use
- HF_TOKEN: Hugging Face token (optional, for HF deployment)
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

from env import OpsFlowEnv
from models import Action
from tasks import list_tasks, get_task


# Load environment variables
load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MAX_STEPS_PER_TASK = 15
MAX_RETRIES = 3


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an AI agent operating a customer support workflow system. Your job is to resolve customer tickets by using the available tools in the correct order.

AVAILABLE TOOLS:
1. READ_TICKET - Read the customer support ticket (always do this first)
2. GET_ORDER_DETAILS - Retrieve order information from the database
3. GET_CUSTOMER_PROFILE - Get customer profile and tier information
4. CHECK_POLICY - Check the applicable refund/return policy
5. REQUEST_APPROVAL - Request manager approval (required for high-value refunds)
6. EXECUTE_REFUND - Process a refund for an order
7. ISSUE_STORE_CREDIT - Issue store credit to customer account
8. SEND_CUSTOMER_REPLY - Send a response message to the customer
9. SUBMIT_RESOLUTION - Submit the final resolution and close the ticket

WORKFLOW RULES:
- Always READ_TICKET first to understand the request
- GET_ORDER_DETAILS before making any decisions about refunds
- CHECK_POLICY before executing refunds to verify eligibility
- REQUEST_APPROVAL is required for refunds above the policy threshold
- Always SEND_CUSTOMER_REPLY before SUBMIT_RESOLUTION
- End with SUBMIT_RESOLUTION to close the ticket

IMPORTANT:
- Follow the correct order of operations
- Don't skip required steps (like policy check before refund)
- Request approval when the refund amount exceeds the threshold
- Always communicate the outcome to the customer

Respond with a JSON object containing:
{
    "tool_name": "TOOL_NAME",
    "arguments": {"key": "value"},
    "reasoning": "Brief explanation of why you chose this action"
}

Only respond with the JSON object, no other text."""


# =============================================================================
# Agent Class
# =============================================================================

class OpsFlowAgent:
    """LLM-based agent for the OpsFlow environment."""
    
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
    
    def reset(self):
        """Reset conversation history for new episode."""
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    
    def get_action(self, observation: Dict[str, Any]) -> Action:
        """
        Get next action from the LLM based on current observation.
        
        Args:
            observation: Current environment observation
            
        Returns:
            Action to execute
        """
        # Build observation message
        obs_text = self._format_observation(observation)
        self.conversation_history.append({"role": "user", "content": obs_text})
        
        # Call LLM
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=0.0,  # Deterministic
                    max_tokens=500
                )
                
                assistant_message = response.choices[0].message.content.strip()
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                # Parse action from response
                action = self._parse_action(assistant_message)
                return action
                
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                    continue
                else:
                    # Fallback action
                    return Action(
                        tool_name="SUBMIT_RESOLUTION",
                        arguments={"status": "failed", "summary": f"Error: {str(e)}"},
                        reasoning="Fallback due to API error"
                    )
    
    def _format_observation(self, observation: Dict[str, Any]) -> str:
        """Format observation as text for the LLM."""
        parts = [
            f"Current Status: {observation.get('current_status', 'unknown')}",
            f"Steps Remaining: {observation.get('max_steps_remaining', 0)}",
            f"\nTicket:\n{observation.get('ticket_text', 'No ticket loaded')}",
        ]
        
        if observation.get('last_tool_output'):
            parts.append(f"\nLast Tool Output:\n{json.dumps(observation['last_tool_output'], indent=2)}")
        
        if observation.get('workflow_history'):
            history = [f"- Step {h['step']}: {h['tool']}" for h in observation['workflow_history']]
            parts.append(f"\nWorkflow History:\n" + "\n".join(history))
        
        if observation.get('compliance_alerts'):
            parts.append(f"\nCompliance Alerts: {', '.join(observation['compliance_alerts'])}")
        
        parts.append("\nWhat tool should be used next? Respond with JSON only.")
        
        return "\n".join(parts)
    
    def _parse_action(self, response: str) -> Action:
        """Parse LLM response into an Action."""
        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            data = json.loads(response.strip())
            
            return Action(
                tool_name=data.get("tool_name", "SUBMIT_RESOLUTION"),
                arguments=data.get("arguments", {}),
                reasoning=data.get("reasoning")
            )
        except json.JSONDecodeError:
            # Try to extract tool name from text
            for tool in OpsFlowEnv.AVAILABLE_TOOLS:
                if tool in response.upper():
                    return Action(
                        tool_name=tool,
                        arguments={},
                        reasoning="Extracted from non-JSON response"
                    )
            
            # Fallback
            return Action(
                tool_name="SUBMIT_RESOLUTION",
                arguments={"status": "failed"},
                reasoning="Could not parse response"
            )


# =============================================================================
# Logging Functions (Required Format)
# =============================================================================

def log_start(task_id: str, task_info: Dict[str, Any]):
    """Log task start in required format."""
    log_data = {
        "task_id": task_id,
        "difficulty": task_info.get("difficulty", "unknown"),
        "description": task_info.get("description", ""),
        "timestamp": datetime.now().isoformat()
    }
    print(f"[START] {json.dumps(log_data)}")


def log_step(step_num: int, action: Action, reward: float, done: bool, info: Dict[str, Any]):
    """Log step in required format."""
    log_data = {
        "step": step_num,
        "action": {
            "tool_name": action.tool_name,
            "arguments": action.arguments
        },
        "reward": round(reward, 4),
        "done": done,
        "cumulative_reward": round(info.get("total_reward", 0.0), 4),
        "timestamp": datetime.now().isoformat()
    }
    print(f"[STEP] {json.dumps(log_data)}")


def log_end(task_id: str, final_score: float, steps_taken: int, total_reward: float):
    """Log task end in required format."""
    log_data = {
        "task_id": task_id,
        "final_score": round(final_score, 4),
        "steps_taken": steps_taken,
        "total_reward": round(total_reward, 4),
        "timestamp": datetime.now().isoformat()
    }
    print(f"[END] {json.dumps(log_data)}")


# =============================================================================
# Main Inference Loop
# =============================================================================

def run_task(env: OpsFlowEnv, agent: OpsFlowAgent, task_id: str) -> Dict[str, Any]:
    """
    Run a single task and return results.
    
    Args:
        env: The environment instance
        agent: The agent instance
        task_id: Task to run
        
    Returns:
        Results dictionary with score, steps, etc.
    """
    # Get task info
    task_info = env.get_task_info(task_id)
    
    # Log start
    log_start(task_id, task_info)
    
    # Reset environment and agent
    observation = env.reset(task_id=task_id)
    agent.reset()
    
    # Run episode
    done = False
    step_num = 0
    total_reward = 0.0
    
    while not done and step_num < MAX_STEPS_PER_TASK:
        # Get action from agent
        obs_dict = observation.model_dump()
        action = agent.get_action(obs_dict)
        
        # Execute action
        result = env.step(action)
        
        step_num += 1
        total_reward += result.reward
        
        # Log step
        log_step(step_num, action, result.reward, result.done, result.info)
        
        # Update for next iteration
        observation = result.observation
        done = result.done
    
    # Get final score
    final_score = result.info.get("final_score", 0.0) if done else 0.0
    
    # Log end
    log_end(task_id, final_score, step_num, total_reward)
    
    return {
        "task_id": task_id,
        "final_score": final_score,
        "steps_taken": step_num,
        "total_reward": total_reward,
        "success": done and final_score > 0.5
    }


def main():
    """Main inference entry point."""
    # Validate configuration
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY or API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    print(f"# OpenEnv-OpsFlow Inference")
    print(f"# Model: {MODEL_NAME}")
    print(f"# API Base: {API_BASE_URL}")
    print(f"# Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL
    )
    
    # Initialize environment and agent
    env = OpsFlowEnv()
    agent = OpsFlowAgent(client=client, model=MODEL_NAME)
    
    # Get all tasks
    task_ids = list_tasks()
    
    # Run all tasks
    results = []
    for task_id in task_ids:
        print(f"\n# Running task: {task_id}")
        result = run_task(env, agent, task_id)
        results.append(result)
        print()
    
    # Print summary
    print("\n# ============================================")
    print("# SUMMARY")
    print("# ============================================")
    
    total_score = 0.0
    for result in results:
        status = "PASS" if result["success"] else "FAIL"
        print(f"# {result['task_id']}: score={result['final_score']:.4f}, steps={result['steps_taken']}, status={status}")
        total_score += result["final_score"]
    
    avg_score = total_score / len(results) if results else 0.0
    print(f"# Average Score: {avg_score:.4f}")
    print("# ============================================")
    
    # Return exit code based on success
    all_passed = all(r["success"] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
