"""
OpenEnv-OpsFlow: Reward System

Dense reward shaping with partial progress signals.
Penalizes clearly undesirable behavior.
"""

from typing import List, Dict, Any, Set
from models import RewardBreakdown


class RewardCalculator:
    """
    Calculates rewards based on agent actions and workflow correctness.
    
    Reward ranges from -1.0 to 1.0 per step.
    Total episode reward is clamped to [0.0, 1.0] for grading.
    """
    
    # Positive rewards for correct actions
    REWARD_TICKET_READ = 0.05
    REWARD_ORDER_RETRIEVED = 0.10
    REWARD_CUSTOMER_RETRIEVED = 0.08
    REWARD_POLICY_CHECKED = 0.10
    REWARD_APPROVAL_REQUESTED = 0.15
    REWARD_REFUND_EXECUTED = 0.20
    REWARD_CREDIT_ISSUED = 0.15
    REWARD_REPLY_SENT = 0.10
    REWARD_RESOLUTION_SUBMITTED = 0.30
    
    # Penalties for bad actions
    PENALTY_DUPLICATE_CALL = -0.05
    PENALTY_IRRELEVANT_TOOL = -0.10
    PENALTY_WRONG_ORDER = -0.15
    PENALTY_POLICY_VIOLATION = -0.30
    PENALTY_APPROVAL_BYPASS = -0.40
    PENALTY_INVALID_ACTION = -0.20
    PENALTY_PER_EXTRA_STEP = -0.02
    
    def __init__(self, expected_workflow: List[str], requires_approval: bool = False):
        """
        Initialize reward calculator.
        
        Args:
            expected_workflow: List of expected tool names in optimal order
            requires_approval: Whether this task requires manager approval
        """
        self.expected_workflow = expected_workflow
        self.requires_approval = requires_approval
        self.optimal_steps = len(expected_workflow)
        self.tools_called: Set[str] = set()
        self.action_sequence: List[str] = []
        self.total_reward = 0.0
        self.reward_breakdown = RewardBreakdown()
    
    def reset(self):
        """Reset the calculator for a new episode."""
        self.tools_called = set()
        self.action_sequence = []
        self.total_reward = 0.0
        self.reward_breakdown = RewardBreakdown()
    
    def calculate_step_reward(
        self,
        tool_name: str,
        tool_output: Dict[str, Any],
        state: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for a single step.
        
        Args:
            tool_name: Name of the tool that was called
            tool_output: Output from the tool execution
            state: Current environment state
            
        Returns:
            Reward value for this step
        """
        reward = 0.0
        
        # Check for duplicate calls
        if tool_name in self.tools_called and tool_name != "SEND_CUSTOMER_REPLY":
            reward += self.PENALTY_DUPLICATE_CALL
            self.reward_breakdown.efficiency_score += self.PENALTY_DUPLICATE_CALL
            self.tools_called.add(tool_name)
            self.action_sequence.append(tool_name)
            self.total_reward += reward
            return reward
        
        # Track the tool call
        self.tools_called.add(tool_name)
        self.action_sequence.append(tool_name)
        
        # Calculate reward based on tool type
        if tool_name == "READ_TICKET":
            if not state.get("ticket_read", False):
                reward += self.REWARD_TICKET_READ
                self.reward_breakdown.tool_score += self.REWARD_TICKET_READ
        
        elif tool_name == "GET_ORDER_DETAILS":
            if state.get("ticket_read", False):
                reward += self.REWARD_ORDER_RETRIEVED
                self.reward_breakdown.tool_score += self.REWARD_ORDER_RETRIEVED
            else:
                reward += self.PENALTY_WRONG_ORDER
                self.reward_breakdown.order_score += self.PENALTY_WRONG_ORDER
        
        elif tool_name == "GET_CUSTOMER_PROFILE":
            if state.get("ticket_read", False):
                reward += self.REWARD_CUSTOMER_RETRIEVED
                self.reward_breakdown.tool_score += self.REWARD_CUSTOMER_RETRIEVED
            else:
                reward += self.PENALTY_WRONG_ORDER
                self.reward_breakdown.order_score += self.PENALTY_WRONG_ORDER
        
        elif tool_name == "CHECK_POLICY":
            if state.get("order_retrieved"):
                reward += self.REWARD_POLICY_CHECKED
                self.reward_breakdown.compliance_score += self.REWARD_POLICY_CHECKED
            else:
                reward += self.PENALTY_WRONG_ORDER
                self.reward_breakdown.order_score += self.PENALTY_WRONG_ORDER
        
        elif tool_name == "REQUEST_APPROVAL":
            if state.get("policy_checked", False):
                reward += self.REWARD_APPROVAL_REQUESTED
                self.reward_breakdown.compliance_score += self.REWARD_APPROVAL_REQUESTED
            else:
                reward += self.PENALTY_WRONG_ORDER
                self.reward_breakdown.order_score += self.PENALTY_WRONG_ORDER
        
        elif tool_name == "EXECUTE_REFUND":
            # Check if approval was required and obtained
            if self.requires_approval:
                if state.get("approval_status") == "approved":
                    reward += self.REWARD_REFUND_EXECUTED
                    self.reward_breakdown.outcome_score += self.REWARD_REFUND_EXECUTED
                elif not state.get("approval_requested", False):
                    # Bypassed approval - major violation
                    reward += self.PENALTY_APPROVAL_BYPASS
                    self.reward_breakdown.compliance_score += self.PENALTY_APPROVAL_BYPASS
                else:
                    # Approval requested but not approved yet
                    reward += self.PENALTY_POLICY_VIOLATION
                    self.reward_breakdown.compliance_score += self.PENALTY_POLICY_VIOLATION
            else:
                # No approval required
                if state.get("policy_checked", False) or "CHECK_POLICY" not in self.expected_workflow:
                    reward += self.REWARD_REFUND_EXECUTED
                    self.reward_breakdown.outcome_score += self.REWARD_REFUND_EXECUTED
                else:
                    # Should have checked policy first
                    reward += self.PENALTY_WRONG_ORDER
                    self.reward_breakdown.order_score += self.PENALTY_WRONG_ORDER
        
        elif tool_name == "ISSUE_STORE_CREDIT":
            reward += self.REWARD_CREDIT_ISSUED
            self.reward_breakdown.outcome_score += self.REWARD_CREDIT_ISSUED
        
        elif tool_name == "SEND_CUSTOMER_REPLY":
            if tool_output.get("success", False):
                reward += self.REWARD_REPLY_SENT
                self.reward_breakdown.outcome_score += self.REWARD_REPLY_SENT
            else:
                reward += self.PENALTY_INVALID_ACTION
                self.reward_breakdown.outcome_score += self.PENALTY_INVALID_ACTION
        
        elif tool_name == "SUBMIT_RESOLUTION":
            # Big reward for successful resolution
            if tool_output.get("success", False):
                reward += self.REWARD_RESOLUTION_SUBMITTED
                self.reward_breakdown.outcome_score += self.REWARD_RESOLUTION_SUBMITTED
            else:
                reward += self.PENALTY_INVALID_ACTION
                self.reward_breakdown.outcome_score += self.PENALTY_INVALID_ACTION
        
        else:
            # Unknown tool
            reward += self.PENALTY_IRRELEVANT_TOOL
            self.reward_breakdown.tool_score += self.PENALTY_IRRELEVANT_TOOL
        
        # Efficiency penalty for extra steps
        step_count = len(self.action_sequence)
        if step_count > self.optimal_steps:
            extra_penalty = self.PENALTY_PER_EXTRA_STEP
            reward += extra_penalty
            self.reward_breakdown.efficiency_score += extra_penalty
        
        self.total_reward += reward
        self.reward_breakdown.total_reward = self.total_reward
        
        return reward
    
    def get_final_reward(self) -> RewardBreakdown:
        """Get the final reward breakdown for the episode."""
        return self.reward_breakdown
    
    def get_normalized_score(self) -> float:
        """
        Get normalized score between 0.0 and 1.0.
        
        This is used for grading.
        """
        # Map total_reward to [0, 1] range
        # Maximum possible reward is approximately 1.0 (all positive rewards)
        # Minimum is approximately -1.5 (many penalties)
        normalized = (self.total_reward + 0.5) / 1.5
        return max(0.0, min(1.0, normalized))
