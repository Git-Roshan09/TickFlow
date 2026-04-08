#!/usr/bin/env python3
"""
OpenEnv-OpsFlow: Baseline Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
"""

import os
import sys
import json
import time
import textwrap
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

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "opsflow"
MAX_STEPS = 15
MAX_RETRIES = 3  # Number of API retry attempts
TEMPERATURE = 0.0  # Deterministic for reproducibility
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.5


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an AI agent operating a customer support workflow system. Your job is to resolve customer tickets by using the available tools in the correct sequence.

CRITICAL: You MUST follow this exact workflow sequence - do NOT skip steps!

MANDATORY WORKFLOW SEQUENCE:
1. READ_TICKET - ALWAYS start here to officially read the ticket (required even if you can see ticket content)
2. Analyze the request type and gather necessary data:
   - For order inquiries: GET_ORDER_DETAILS 
   - For account issues: GET_CUSTOMER_PROFILE
   - For refund requests: CHECK_POLICY first
3. Take appropriate action based on findings:
   - If refund needed: REQUEST_APPROVAL (if high value) → EXECUTE_REFUND
   - If store credit needed: ISSUE_STORE_CREDIT  
   - If information needed: provide details
4. SEND_CUSTOMER_REPLY - ALWAYS communicate the outcome to customer
5. SUBMIT_RESOLUTION - Close the ticket (FINAL step only)

AVAILABLE TOOLS:
- READ_TICKET: Read the customer support ticket (MANDATORY FIRST STEP)
- GET_ORDER_DETAILS: Retrieve order information from database  
- GET_CUSTOMER_PROFILE: Get customer profile and tier information
- CHECK_POLICY: Check applicable refund/return policy
- REQUEST_APPROVAL: Request manager approval (required for high-value refunds)
- EXECUTE_REFUND: Process a refund for an order
- ISSUE_STORE_CREDIT: Issue store credit to customer account
- SEND_CUSTOMER_REPLY: Send response message to customer (MANDATORY before closing)
- SUBMIT_RESOLUTION: Close the ticket (FINAL STEP ONLY)

CRITICAL RULES:
❌ NEVER start with SUBMIT_RESOLUTION - this ends the workflow
❌ NEVER skip READ_TICKET - always do this first
❌ NEVER skip SEND_CUSTOMER_REPLY before closing
✅ ALWAYS follow the sequence: READ → ANALYZE → ACT → REPLY → CLOSE

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
        ]
        
        # Only show ticket preview initially - force agent to use READ_TICKET
        if not observation.get('workflow_history'):
            parts.append(f"\n📋 Ticket Preview: New customer support ticket available")
            parts.append("⚠️  Use READ_TICKET tool to view full ticket content")
        else:
            # Show full ticket after READ_TICKET has been used
            parts.append(f"\n📋 Full Ticket:\n{observation.get('ticket_text', 'No ticket loaded')}")
        
        if observation.get('last_tool_output'):
            parts.append(f"\n🔧 Last Tool Output:\n{json.dumps(observation['last_tool_output'], indent=2)}")
        
        if observation.get('workflow_history'):
            history = [f"- Step {h['step']}: {h['tool']}" for h in observation['workflow_history']]
            parts.append(f"\n📊 Workflow History:\n" + "\n".join(history))
        
        if observation.get('compliance_alerts'):
            parts.append(f"\n⚠️  Compliance Alerts: {', '.join(observation['compliance_alerts'])}")
        
        available_tools = observation.get('available_tools', [])
        parts.append(f"\n🛠️  Available Tools: {', '.join(available_tools)}")
        
        parts.append("\n❓ What tool should be used next? Remember to follow the mandatory workflow sequence!")
        parts.append("📝 Respond with JSON only.")
        
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

def log_start(task: str, env: str, model: str) -> None:
    """Log task start in required format."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log step in required format."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log task end in required format."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


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
    # Log start
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    result = None
    
    try:
        # Reset environment and agent
        observation = env.reset(task_id=task_id)
        agent.reset()
        
        # Run episode
        done = False
        
        for step in range(1, MAX_STEPS + 1):
            if done:
                break
            
            # Get action from agent
            obs_dict = observation.model_dump()
            action = agent.get_action(obs_dict)
            
            # Execute action
            result = env.step(action)
            
            reward = result.reward or 0.0
            done = result.done
            error = result.info.get("error") if result.info else None
            
            rewards.append(reward)
            steps_taken = step
            
            # Format action string for logging
            action_str = f"{action.tool_name}({json.dumps(action.arguments)})"
            
            # Log step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            
            # Update for next iteration
            observation = result.observation
            
            if done:
                break
        
        # Get final score from grader
        if done and result and result.info:
            score = result.info.get("final_score", 0.0)
        else:
            # If not done, use normalized cumulative reward
            score = max(0.0, min(1.0, sum(rewards)))
        
        score = max(0.0, min(1.0, score))  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    except Exception as e:
        print(f"[DEBUG] Task {task_id} failed with error: {e}", flush=True)
        score = 0.0
        success = False
    
    finally:
        # Always log end
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return {
        "task_id": task_id,
        "final_score": score,
        "steps_taken": steps_taken,
        "total_reward": sum(rewards),
        "success": success
    }


def main():
    """Main inference entry point."""
    # Validate configuration
    if not API_KEY:
        print("[DEBUG] ERROR: HF_TOKEN or API_KEY environment variable not set", flush=True)
        sys.exit(1)
    
    # Initialize OpenAI client
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Initialize environment and agent
    env = OpsFlowEnv()
    agent = OpsFlowAgent(client=client, model=MODEL_NAME)
    
    # Get all tasks
    task_ids = list_tasks()
    
    # Run all tasks
    results = []
    for task_id in task_ids:
        result = run_task(env, agent, task_id)
        results.append(result)
    
    # Print summary (debug output)
    print(f"\n# SUMMARY", flush=True)
    total_score = sum(r["final_score"] for r in results)
    avg_score = total_score / len(results) if results else 0.0
    print(f"# Average Score: {avg_score:.3f}", flush=True)
    
    # Return exit code based on success
    all_passed = all(r["success"] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
