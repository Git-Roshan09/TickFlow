"""
OpenEnv-OpsFlow: Typed Pydantic Models

This module defines the core data models for the OpsFlow environment:
- Action: What the agent can do
- Observation: What the agent sees
- RewardBreakdown: How rewards are computed
- Internal state models for enterprise simulation
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# Tool Names Enum
# =============================================================================

class ToolName(str, Enum):
    READ_TICKET = "READ_TICKET"
    GET_ORDER_DETAILS = "GET_ORDER_DETAILS"
    GET_CUSTOMER_PROFILE = "GET_CUSTOMER_PROFILE"
    CHECK_POLICY = "CHECK_POLICY"
    REQUEST_APPROVAL = "REQUEST_APPROVAL"
    EXECUTE_REFUND = "EXECUTE_REFUND"
    ISSUE_STORE_CREDIT = "ISSUE_STORE_CREDIT"
    SEND_CUSTOMER_REPLY = "SEND_CUSTOMER_REPLY"
    SUBMIT_RESOLUTION = "SUBMIT_RESOLUTION"


# =============================================================================
# Action Model
# =============================================================================

class Action(BaseModel):
    """
    Represents an action taken by the agent.
    
    The agent selects a tool and provides optional arguments and reasoning.
    """
    tool_name: Literal[
        "READ_TICKET",
        "GET_ORDER_DETAILS",
        "GET_CUSTOMER_PROFILE",
        "CHECK_POLICY",
        "REQUEST_APPROVAL",
        "EXECUTE_REFUND",
        "ISSUE_STORE_CREDIT",
        "SEND_CUSTOMER_REPLY",
        "SUBMIT_RESOLUTION"
    ] = Field(..., description="The tool to execute")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool (e.g., order_id, message)"
    )
    reasoning: Optional[str] = Field(
        None,
        description="Agent's reasoning for this action"
    )


# =============================================================================
# Observation Model
# =============================================================================

class Observation(BaseModel):
    """
    Represents what the agent observes after each step.
    
    Contains task context, tool outputs, workflow history, and constraints.
    """
    task_id: str = Field(..., description="Unique identifier for the current task")
    ticket_text: str = Field(..., description="The customer support ticket content")
    available_tools: List[str] = Field(
        default_factory=list,
        description="List of tools the agent can use"
    )
    last_tool_output: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output from the last tool execution"
    )
    workflow_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of actions taken in this episode"
    )
    compliance_alerts: List[str] = Field(
        default_factory=list,
        description="Any compliance warnings or alerts"
    )
    budget_remaining: float = Field(
        0.0,
        description="Remaining budget for operations (if applicable)"
    )
    max_steps_remaining: int = Field(
        0,
        description="Number of steps remaining before episode ends"
    )
    current_status: str = Field(
        "pending",
        description="Current status of the ticket resolution"
    )


# =============================================================================
# Reward Breakdown Model
# =============================================================================

class RewardBreakdown(BaseModel):
    """
    Detailed breakdown of the reward components.
    
    Used internally for debugging and reward shaping.
    """
    tool_score: float = Field(0.0, description="Score for using correct tools")
    order_score: float = Field(0.0, description="Score for correct action ordering")
    compliance_score: float = Field(0.0, description="Score for policy compliance")
    efficiency_score: float = Field(0.0, description="Score for efficiency (fewer steps)")
    outcome_score: float = Field(0.0, description="Score for final outcome quality")
    total_reward: float = Field(0.0, description="Sum of all reward components")


# =============================================================================
# Step Result Model
# =============================================================================

class StepResult(BaseModel):
    """Result returned by step() method."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Internal State Models
# =============================================================================

class Order(BaseModel):
    """Represents a customer order in the system."""
    order_id: str
    customer_id: str
    product_name: str
    product_category: str
    order_amount: float
    order_date: str
    delivery_status: Literal["pending", "shipped", "delivered", "returned", "cancelled"]
    delivery_date: Optional[str] = None
    tracking_number: Optional[str] = None
    issue_type: Optional[str] = None
    issue_description: Optional[str] = None


class Customer(BaseModel):
    """Represents a customer profile."""
    customer_id: str
    name: str
    email: str
    tier: Literal["standard", "premium", "vip"]
    account_age_days: int
    total_orders: int
    total_spent: float
    has_active_subscription: bool = False
    fraud_flag: bool = False


class Policy(BaseModel):
    """Represents a company policy rule."""
    policy_id: str
    policy_name: str
    description: str
    refund_threshold: float = 100.0
    approval_required_above: float = 100.0
    max_refund_days: int = 30
    vip_credit_bonus_percent: float = 10.0
    fraud_block_enabled: bool = True


class ApprovalRecord(BaseModel):
    """Represents an approval request and its status."""
    approval_id: str
    order_id: str
    requested_amount: float
    reason: str
    status: Literal["pending", "approved", "denied"]
    approver: Optional[str] = None
    timestamp: Optional[str] = None


class RefundRecord(BaseModel):
    """Represents a refund execution record."""
    refund_id: str
    order_id: str
    amount: float
    reason: str
    status: Literal["pending", "completed", "failed"]
    timestamp: str


class CreditRecord(BaseModel):
    """Represents a store credit issuance record."""
    credit_id: str
    customer_id: str
    amount: float
    reason: str
    timestamp: str


# =============================================================================
# Task Definition Model
# =============================================================================

class TaskDefinition(BaseModel):
    """Defines a task scenario for the environment."""
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    ticket_text: str
    description: str
    order_id: str
    customer_id: str
    expected_workflow: List[str] = Field(
        default_factory=list,
        description="Expected optimal sequence of tools"
    )
    requires_approval: bool = False
    requires_policy_check: bool = False
    success_criteria: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Environment State Model
# =============================================================================

class EnvironmentState(BaseModel):
    """Complete state of the environment (for state() method)."""
    task_id: str
    task_difficulty: str
    ticket_text: str
    step_count: int
    max_steps: int
    done: bool
    
    # What has been retrieved/executed
    ticket_read: bool = False
    order_retrieved: Optional[Dict[str, Any]] = None
    customer_retrieved: Optional[Dict[str, Any]] = None
    policy_checked: bool = False
    approval_requested: bool = False
    approval_status: Optional[str] = None
    refund_executed: bool = False
    refund_amount: float = 0.0
    credit_issued: bool = False
    credit_amount: float = 0.0
    customer_reply_sent: bool = False
    customer_reply_content: Optional[str] = None
    resolution_submitted: bool = False
    resolution_status: Optional[str] = None
    
    # Workflow tracking
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    compliance_violations: List[str] = Field(default_factory=list)
    total_reward: float = 0.0
