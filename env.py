"""
OpenEnv-OpsFlow: Main Environment

Implements the OpsFlowEnv class with step(), reset(), state() API.
Simulates enterprise customer support and compliance operations.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from models import (
    Action, Observation, EnvironmentState, StepResult,
    Order, Customer, Policy, ApprovalRecord, RefundRecord, CreditRecord,
    TaskDefinition
)
from rewards import RewardCalculator
from graders import grade_task
from tasks import get_task, list_tasks, TASKS


class OpsFlowEnv:
    """
    Enterprise Customer Support and Compliance Operations Environment.
    
    The agent must resolve support tickets by:
    1. Reading the ticket
    2. Using appropriate internal tools
    3. Following company policy
    4. Requesting approvals when needed
    5. Executing business actions
    6. Communicating with customers
    7. Submitting final resolution
    """
    
    MAX_STEPS = 15
    
    AVAILABLE_TOOLS = [
        "READ_TICKET",
        "GET_ORDER_DETAILS",
        "GET_CUSTOMER_PROFILE",
        "CHECK_POLICY",
        "REQUEST_APPROVAL",
        "EXECUTE_REFUND",
        "ISSUE_STORE_CREDIT",
        "SEND_CUSTOMER_REPLY",
        "SUBMIT_RESOLUTION"
    ]
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the environment.
        
        Args:
            data_dir: Path to data directory. Defaults to ./data/
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.data_dir = Path(data_dir)
        
        # Load mock enterprise data
        self._load_data()
        
        # Initialize state
        self._reset_state()
    
    def _load_data(self):
        """Load mock enterprise data from JSON files."""
        # Load orders
        orders_file = self.data_dir / "orders.json"
        with open(orders_file, "r") as f:
            orders_data = json.load(f)
        self.orders: Dict[str, Order] = {
            o["order_id"]: Order(**o) for o in orders_data
        }
        
        # Load customers
        customers_file = self.data_dir / "customers.json"
        with open(customers_file, "r") as f:
            customers_data = json.load(f)
        self.customers: Dict[str, Customer] = {
            c["customer_id"]: Customer(**c) for c in customers_data
        }
        
        # Load policies
        policies_file = self.data_dir / "policies.json"
        with open(policies_file, "r") as f:
            policies_data = json.load(f)
        self.policies: Dict[str, Policy] = {
            p["policy_id"]: Policy(**p) for p in policies_data
        }
    
    def _reset_state(self):
        """Reset all internal state."""
        self.current_task: Optional[TaskDefinition] = None
        self.step_count = 0
        self.done = False
        
        # Workflow state
        self.ticket_read = False
        self.order_retrieved: Optional[Dict[str, Any]] = None
        self.customer_retrieved: Optional[Dict[str, Any]] = None
        self.policy_checked = False
        self.current_policy: Optional[Policy] = None
        self.approval_requested = False
        self.approval_status: Optional[str] = None
        self.refund_executed = False
        self.refund_amount = 0.0
        self.credit_issued = False
        self.credit_amount = 0.0
        self.customer_reply_sent = False
        self.customer_reply_content: Optional[str] = None
        self.resolution_submitted = False
        self.resolution_status: Optional[str] = None
        
        # Tracking
        self.action_history: List[Dict[str, Any]] = []
        self.compliance_violations: List[str] = []
        self.total_reward = 0.0
        
        # Reward calculator
        self.reward_calculator: Optional[RewardCalculator] = None
    
    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Reset the environment to a fresh state.
        
        Args:
            task_id: Optional task to load. If None, loads the first task.
            
        Returns:
            Initial observation
        """
        self._reset_state()
        
        # Load task
        if task_id is None:
            task_id = list_tasks()[0]
        
        self.current_task = get_task(task_id)
        
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(
            expected_workflow=self.current_task.expected_workflow,
            requires_approval=self.current_task.requires_approval
        )
        
        return self._get_observation()
    
    def step(self, action: Action) -> StepResult:
        """
        Execute an action in the environment.
        
        Args:
            action: The action to execute
            
        Returns:
            StepResult containing observation, reward, done, info
        """
        if self.done:
            return StepResult(
                observation=self._get_observation(),
                reward=0.0,
                done=True,
                info={"error": "Episode already finished. Call reset() to start a new episode."}
            )
        
        if self.current_task is None:
            return StepResult(
                observation=self._get_observation(),
                reward=0.0,
                done=True,
                info={"error": "No task loaded. Call reset() first."}
            )
        
        # Increment step counter
        self.step_count += 1
        
        # Execute the tool
        tool_output = self._execute_tool(action)
        
        # Record action in history
        self.action_history.append({
            "step": self.step_count,
            "tool_name": action.tool_name,
            "arguments": action.arguments,
            "reasoning": action.reasoning,
            "output": tool_output,
            "timestamp": datetime.now().isoformat()
        })
        
        # Calculate reward
        state_dict = self._get_state_dict()
        reward = self.reward_calculator.calculate_step_reward(
            action.tool_name,
            tool_output,
            state_dict
        )
        self.total_reward += reward
        
        # Check for terminal conditions
        if action.tool_name == "SUBMIT_RESOLUTION":
            self.done = True
        elif self.step_count >= self.MAX_STEPS:
            self.done = True
            self.compliance_violations.append("max_steps_exceeded")
        
        # Build info dict
        info = {
            "tool_output": tool_output,
            "step_reward": reward,
            "total_reward": self.total_reward,
            "steps_taken": self.step_count,
            "reward_breakdown": self.reward_calculator.get_final_reward().model_dump()
        }
        
        # Add final score if done
        if self.done:
            final_state = self.state()
            final_score = grade_task(self.current_task.task_id, final_state)
            info["final_score"] = final_score
            info["grader_score"] = final_score
        
        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=self.done,
            info=info
        )
    
    def state(self) -> EnvironmentState:
        """
        Get the current complete environment state.
        
        Returns:
            EnvironmentState with all internal state
        """
        return EnvironmentState(
            task_id=self.current_task.task_id if self.current_task else "",
            task_difficulty=self.current_task.difficulty if self.current_task else "",
            ticket_text=self.current_task.ticket_text if self.current_task else "",
            step_count=self.step_count,
            max_steps=self.MAX_STEPS,
            done=self.done,
            ticket_read=self.ticket_read,
            order_retrieved=self.order_retrieved,
            customer_retrieved=self.customer_retrieved,
            policy_checked=self.policy_checked,
            approval_requested=self.approval_requested,
            approval_status=self.approval_status,
            refund_executed=self.refund_executed,
            refund_amount=self.refund_amount,
            credit_issued=self.credit_issued,
            credit_amount=self.credit_amount,
            customer_reply_sent=self.customer_reply_sent,
            customer_reply_content=self.customer_reply_content,
            resolution_submitted=self.resolution_submitted,
            resolution_status=self.resolution_status,
            action_history=self.action_history,
            compliance_violations=self.compliance_violations,
            total_reward=self.total_reward
        )
    
    def _get_observation(self) -> Observation:
        """Build current observation for the agent."""
        last_output = {}
        if self.action_history:
            last_output = self.action_history[-1].get("output", {})
        
        return Observation(
            task_id=self.current_task.task_id if self.current_task else "",
            ticket_text=self.current_task.ticket_text if self.current_task else "",
            available_tools=self.AVAILABLE_TOOLS,
            last_tool_output=last_output,
            workflow_history=[
                {"step": a["step"], "tool": a["tool_name"]}
                for a in self.action_history
            ],
            compliance_alerts=self.compliance_violations.copy(),
            budget_remaining=1000.0,  # Simplified budget
            max_steps_remaining=self.MAX_STEPS - self.step_count,
            current_status=self._get_current_status()
        )
    
    def _get_current_status(self) -> str:
        """Get human-readable current status."""
        if self.resolution_submitted:
            return f"resolved_{self.resolution_status}"
        if self.customer_reply_sent:
            return "reply_sent"
        if self.refund_executed:
            return "refund_executed"
        if self.credit_issued:
            return "credit_issued"
        if self.approval_status == "approved":
            return "approval_obtained"
        if self.approval_requested:
            return "awaiting_approval"
        if self.policy_checked:
            return "policy_checked"
        if self.order_retrieved:
            return "order_retrieved"
        if self.ticket_read:
            return "ticket_read"
        return "pending"
    
    def _get_state_dict(self) -> Dict[str, Any]:
        """Get state as dictionary for reward calculation."""
        return {
            "ticket_read": self.ticket_read,
            "order_retrieved": self.order_retrieved,
            "customer_retrieved": self.customer_retrieved,
            "policy_checked": self.policy_checked,
            "approval_requested": self.approval_requested,
            "approval_status": self.approval_status,
            "refund_executed": self.refund_executed,
            "credit_issued": self.credit_issued,
        }
    
    def _execute_tool(self, action: Action) -> Dict[str, Any]:
        """
        Execute a tool and return its output.
        
        Args:
            action: The action containing tool name and arguments
            
        Returns:
            Tool output dictionary
        """
        tool_name = action.tool_name
        args = action.arguments
        
        if tool_name == "READ_TICKET":
            return self._tool_read_ticket()
        elif tool_name == "GET_ORDER_DETAILS":
            return self._tool_get_order_details(args.get("order_id"))
        elif tool_name == "GET_CUSTOMER_PROFILE":
            return self._tool_get_customer_profile(args.get("customer_id"))
        elif tool_name == "CHECK_POLICY":
            return self._tool_check_policy(args.get("customer_tier"))
        elif tool_name == "REQUEST_APPROVAL":
            return self._tool_request_approval(
                args.get("order_id"),
                args.get("amount"),
                args.get("reason", "Refund request")
            )
        elif tool_name == "EXECUTE_REFUND":
            return self._tool_execute_refund(
                args.get("order_id"),
                args.get("amount"),
                args.get("reason", "Customer request")
            )
        elif tool_name == "ISSUE_STORE_CREDIT":
            return self._tool_issue_store_credit(
                args.get("customer_id"),
                args.get("amount"),
                args.get("reason", "Compensation")
            )
        elif tool_name == "SEND_CUSTOMER_REPLY":
            return self._tool_send_customer_reply(args.get("message", ""))
        elif tool_name == "SUBMIT_RESOLUTION":
            return self._tool_submit_resolution(
                args.get("status", "resolved"),
                args.get("summary", "")
            )
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    # ==========================================================================
    # Tool Implementations
    # ==========================================================================
    
    def _tool_read_ticket(self) -> Dict[str, Any]:
        """Read the current support ticket."""
        if not self.current_task:
            return {"success": False, "error": "No ticket loaded"}
        
        self.ticket_read = True
        
        return {
            "success": True,
            "ticket_id": self.current_task.task_id,
            "ticket_text": self.current_task.ticket_text,
            "order_id": self.current_task.order_id,
            "customer_id": self.current_task.customer_id,
            "detected_intent": self._detect_intent(self.current_task.ticket_text)
        }
    
    def _detect_intent(self, ticket_text: str) -> str:
        """Simple intent detection from ticket text."""
        text_lower = ticket_text.lower()
        if "refund" in text_lower:
            return "refund_request"
        if "where" in text_lower and "order" in text_lower:
            return "delivery_inquiry"
        if "damaged" in text_lower or "defective" in text_lower:
            return "product_issue"
        if "exchange" in text_lower or "wrong" in text_lower:
            return "exchange_request"
        return "general_inquiry"
    
    def _tool_get_order_details(self, order_id: Optional[str] = None) -> Dict[str, Any]:
        """Get order details from the system."""
        if not self.ticket_read:
            self.compliance_violations.append("order_lookup_before_ticket_read")
        
        # Use order_id from task if not provided
        if order_id is None and self.current_task:
            order_id = self.current_task.order_id
        
        if order_id is None or order_id not in self.orders:
            return {"success": False, "error": f"Order not found: {order_id}"}
        
        order = self.orders[order_id]
        self.order_retrieved = order.model_dump()
        
        return {
            "success": True,
            "order": order.model_dump()
        }
    
    def _tool_get_customer_profile(self, customer_id: Optional[str] = None) -> Dict[str, Any]:
        """Get customer profile from the system."""
        if not self.ticket_read:
            self.compliance_violations.append("customer_lookup_before_ticket_read")
        
        # Use customer_id from task if not provided
        if customer_id is None and self.current_task:
            customer_id = self.current_task.customer_id
        
        if customer_id is None or customer_id not in self.customers:
            return {"success": False, "error": f"Customer not found: {customer_id}"}
        
        customer = self.customers[customer_id]
        
        # Check for fraud flag
        if customer.fraud_flag:
            self.compliance_violations.append("fraud_flag_detected")
        
        self.customer_retrieved = customer.model_dump()
        
        return {
            "success": True,
            "customer": customer.model_dump()
        }
    
    def _tool_check_policy(self, customer_tier: Optional[str] = None) -> Dict[str, Any]:
        """Check applicable refund/return policy."""
        # Determine which policy applies
        if customer_tier is None and self.customer_retrieved:
            customer_tier = self.customer_retrieved.get("tier", "standard")
        
        if customer_tier is None:
            customer_tier = "standard"
        
        # Map tier to policy
        policy_map = {
            "standard": "POL-001",
            "premium": "POL-002",
            "vip": "POL-003"
        }
        
        policy_id = policy_map.get(customer_tier, "POL-001")
        policy = self.policies.get(policy_id)
        
        if policy is None:
            return {"success": False, "error": "Policy not found"}
        
        self.policy_checked = True
        self.current_policy = policy
        
        # Determine if approval is needed
        order_amount = 0.0
        if self.order_retrieved:
            order_amount = self.order_retrieved.get("order_amount", 0.0)
        
        approval_required = order_amount > policy.approval_required_above
        
        return {
            "success": True,
            "policy": policy.model_dump(),
            "order_amount": order_amount,
            "approval_required": approval_required,
            "max_refund_days": policy.max_refund_days,
            "notes": f"{'Manager approval required' if approval_required else 'Auto-approval allowed'}"
        }
    
    def _tool_request_approval(
        self,
        order_id: Optional[str] = None,
        amount: Optional[float] = None,
        reason: str = "Refund request"
    ) -> Dict[str, Any]:
        """Request manager approval for an action."""
        if not self.policy_checked:
            self.compliance_violations.append("approval_without_policy_check")
        
        if order_id is None and self.order_retrieved:
            order_id = self.order_retrieved.get("order_id")
        
        if amount is None and self.order_retrieved:
            amount = self.order_retrieved.get("order_amount", 0.0)
        
        if order_id is None:
            return {"success": False, "error": "No order specified for approval"}
        
        self.approval_requested = True
        
        # Simulate approval process (auto-approve for valid requests)
        # In a real system, this would involve human review
        self.approval_status = "approved"
        
        return {
            "success": True,
            "approval_id": f"APR-{order_id}",
            "status": "approved",
            "amount": amount,
            "reason": reason,
            "approver": "Manager-AutoApprove",
            "timestamp": datetime.now().isoformat(),
            "notes": "Approval granted for refund/credit operation"
        }
    
    def _tool_execute_refund(
        self,
        order_id: Optional[str] = None,
        amount: Optional[float] = None,
        reason: str = "Customer request"
    ) -> Dict[str, Any]:
        """Execute a refund for an order."""
        if order_id is None and self.order_retrieved:
            order_id = self.order_retrieved.get("order_id")
        
        if amount is None and self.order_retrieved:
            amount = self.order_retrieved.get("order_amount", 0.0)
        
        if order_id is None:
            return {"success": False, "error": "No order specified for refund"}
        
        # Check if approval was required but not obtained
        if self.current_task and self.current_task.requires_approval:
            if not self.approval_requested:
                self.compliance_violations.append("refund_without_required_approval")
            elif self.approval_status != "approved":
                self.compliance_violations.append("refund_with_pending_approval")
        
        # Check for fraud
        if self.customer_retrieved and self.customer_retrieved.get("fraud_flag"):
            self.compliance_violations.append("refund_to_fraud_flagged_customer")
            return {
                "success": False,
                "error": "Refund blocked: Customer has fraud flag",
                "blocked_by": "fraud_detection"
            }
        
        self.refund_executed = True
        self.refund_amount = amount or 0.0
        
        return {
            "success": True,
            "refund_id": f"REF-{order_id}",
            "order_id": order_id,
            "amount": amount,
            "reason": reason,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "notes": "Refund has been processed successfully"
        }
    
    def _tool_issue_store_credit(
        self,
        customer_id: Optional[str] = None,
        amount: Optional[float] = None,
        reason: str = "Compensation"
    ) -> Dict[str, Any]:
        """Issue store credit to a customer."""
        if customer_id is None and self.customer_retrieved:
            customer_id = self.customer_retrieved.get("customer_id")
        
        if customer_id is None:
            return {"success": False, "error": "No customer specified for credit"}
        
        # Apply VIP bonus if applicable
        bonus_percent = 0.0
        if self.current_policy:
            bonus_percent = self.current_policy.vip_credit_bonus_percent
        
        final_amount = amount or 0.0
        if self.customer_retrieved and self.customer_retrieved.get("tier") == "vip":
            final_amount = final_amount * (1 + bonus_percent / 100)
        
        self.credit_issued = True
        self.credit_amount = final_amount
        
        return {
            "success": True,
            "credit_id": f"CRD-{customer_id}",
            "customer_id": customer_id,
            "amount": final_amount,
            "original_amount": amount,
            "bonus_applied": bonus_percent if self.customer_retrieved and self.customer_retrieved.get("tier") == "vip" else 0.0,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "notes": "Store credit has been added to customer account"
        }
    
    def _tool_send_customer_reply(self, message: str) -> Dict[str, Any]:
        """Send a reply to the customer."""
        if not message or len(message.strip()) < 10:
            return {
                "success": False,
                "error": "Message too short. Please provide a meaningful response."
            }
        
        self.customer_reply_sent = True
        self.customer_reply_content = message
        
        return {
            "success": True,
            "message_sent": True,
            "message_preview": message[:100] + "..." if len(message) > 100 else message,
            "timestamp": datetime.now().isoformat(),
            "notes": "Customer has been notified via email"
        }
    
    def _tool_submit_resolution(
        self,
        status: str = "resolved",
        summary: str = ""
    ) -> Dict[str, Any]:
        """Submit the final resolution for the ticket."""
        self.resolution_submitted = True
        self.resolution_status = status
        
        # Validate resolution
        warnings = []
        if not self.ticket_read:
            warnings.append("Ticket was never read")
        if not self.customer_reply_sent:
            warnings.append("No customer reply was sent")
        
        return {
            "success": True,
            "resolution_id": f"RES-{self.current_task.task_id if self.current_task else 'unknown'}",
            "status": status,
            "summary": summary,
            "steps_taken": self.step_count,
            "timestamp": datetime.now().isoformat(),
            "warnings": warnings,
            "notes": "Ticket has been closed"
        }
    
    # ==========================================================================
    # Utility Methods
    # ==========================================================================
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available task IDs."""
        return list_tasks()
    
    def get_task_info(self, task_id: str) -> Dict[str, Any]:
        """Get information about a specific task."""
        task = get_task(task_id)
        return {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "description": task.description,
            "expected_steps": len(task.expected_workflow),
            "requires_approval": task.requires_approval
        }
