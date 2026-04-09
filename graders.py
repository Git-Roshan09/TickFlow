"""
OpenEnv-OpsFlow: Graders

Deterministic graders for each task.
Returns scores between 0.0 and 1.0.
"""

from typing import Dict, Any, List, Optional
from models import EnvironmentState

EPSILON = 1e-3


def _to_open_unit_interval(score: float) -> float:
    """Clamp score to strict open interval (0, 1)."""
    return max(EPSILON, min(1.0 - EPSILON, score))


class BaseGrader:
    """Base class for task graders."""
    
    def grade(self, state: EnvironmentState) -> float:
        """
        Grade the final state.
        
        Args:
            state: The final environment state
            
        Returns:
            Score between 0.0 and 1.0
        """
        raise NotImplementedError


class DeliveryStatusGrader(BaseGrader):
    """
    Grader for Task 1: Delivery Status Resolution (Easy)
    
    Success criteria:
    - 1.0: Correct order status retrieved and communicated
    - 0.5: Status retrieved but reply incomplete
    - 0.0: Agent hallucinates or fails
    """
    
    def grade(self, state: EnvironmentState) -> float:
        score = 0.0
        
        # Check if order was retrieved
        if state.order_retrieved:
            score += 0.3
        
        # Check if customer reply was sent
        if state.customer_reply_sent:
            score += 0.2
            # Check if reply contains relevant information
            if state.customer_reply_content:
                reply_lower = state.customer_reply_content.lower()
                if any(word in reply_lower for word in ["shipped", "tracking", "delivery", "order"]):
                    score += 0.2
        
        # Check if resolution was submitted
        if state.resolution_submitted:
            score += 0.3
        
        # Penalty for compliance violations
        violation_penalty = len(state.compliance_violations) * 0.1
        score = max(0.0, score - violation_penalty)
        
        return _to_open_unit_interval(score)


class LowValueRefundGrader(BaseGrader):
    """
    Grader for Task 2: Low-Value Refund with Policy Check (Medium)
    
    Success criteria:
    - 1.0: Refund executed after proper policy check
    - 0.7: Refund correct but reply weak
    - 0.5: Refund executed without policy check
    - 0.0: Denied incorrectly or failed
    """
    
    def grade(self, state: EnvironmentState) -> float:
        score = 0.0
        
        # Check order retrieval
        if state.order_retrieved:
            score += 0.15
        
        # Check policy was checked (important for this task)
        if state.policy_checked:
            score += 0.20
        
        # Check refund was executed
        if state.refund_executed:
            if state.policy_checked:
                score += 0.25  # Full points if policy was checked
            else:
                score += 0.10  # Partial points if bypassed policy
        
        # Check customer reply
        if state.customer_reply_sent:
            score += 0.15
            if state.customer_reply_content:
                reply_lower = state.customer_reply_content.lower()
                if any(word in reply_lower for word in ["refund", "processed", "approved", "amount"]):
                    score += 0.10
        
        # Check resolution
        if state.resolution_submitted:
            score += 0.15
        
        # Penalty for compliance violations
        violation_penalty = len(state.compliance_violations) * 0.15
        score = max(0.0, score - violation_penalty)
        
        return _to_open_unit_interval(score)


class HighValueApprovalGrader(BaseGrader):
    """
    Grader for Task 3: High-Value Refund Requiring Approval (Hard)
    
    Success criteria:
    - 1.0: Full approval flow correct
    - 0.6: Workflow mostly correct but messaging weak
    - 0.2: Refund happens after unsafe sequence
    - 0.0: Approval bypassed or task fails
    """
    
    def grade(self, state: EnvironmentState) -> float:
        score = 0.0
        
        # Check order retrieval
        if state.order_retrieved:
            score += 0.10
        
        # Check customer profile retrieval (important for VIP handling)
        if state.customer_retrieved:
            score += 0.10
        
        # Check policy was checked
        if state.policy_checked:
            score += 0.15
        
        # Critical: Check if approval was requested
        if state.approval_requested:
            score += 0.15
            # Check if approval was obtained before refund
            if state.approval_status == "approved":
                score += 0.10
        
        # Check refund execution
        if state.refund_executed:
            if state.approval_requested and state.approval_status == "approved":
                score += 0.15  # Full points - proper approval flow
            elif state.approval_requested:
                score += 0.05  # Partial - at least tried
            else:
                # Major penalty - bypassed approval entirely
                score -= 0.30
        
        # Check customer reply
        if state.customer_reply_sent:
            score += 0.10
            if state.customer_reply_content:
                reply_lower = state.customer_reply_content.lower()
                if any(word in reply_lower for word in ["refund", "approved", "vip", "processed"]):
                    score += 0.05
        
        # Check resolution
        if state.resolution_submitted:
            score += 0.10
        
        # Heavy penalty for compliance violations
        violation_penalty = len(state.compliance_violations) * 0.20
        score = max(0.0, score - violation_penalty)
        
        return _to_open_unit_interval(score)


# =============================================================================
# Grader Registry
# =============================================================================

GRADERS = {
    "task_easy_delivery": DeliveryStatusGrader(),
    "task_medium_refund": LowValueRefundGrader(),
    "task_hard_approval": HighValueApprovalGrader(),
}


def grade_task(task_id: str, state: EnvironmentState) -> float:
    """
    Grade a completed task.
    
    Args:
        task_id: The task identifier
        state: The final environment state
        
    Returns:
        Score between 0.0 and 1.0
    """
    if task_id not in GRADERS:
        raise ValueError(f"No grader for task: {task_id}")
    
    return _to_open_unit_interval(GRADERS[task_id].grade(state))


def get_grader(task_id: str) -> BaseGrader:
    """Get grader instance for a task."""
    if task_id not in GRADERS:
        raise ValueError(f"No grader for task: {task_id}")
    return GRADERS[task_id]
