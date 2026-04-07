"""
OpenEnv-OpsFlow: Test Suite for Environment
"""

import pytest
from env import OpsFlowEnv
from models import Action, Observation, EnvironmentState
from tasks import list_tasks


class TestEnvironmentReset:
    """Tests for environment reset functionality."""
    
    def test_reset_returns_observation(self):
        """Reset should return an Observation object."""
        env = OpsFlowEnv()
        obs = env.reset()
        assert isinstance(obs, Observation)
    
    def test_reset_clears_state(self):
        """Reset should clear all internal state."""
        env = OpsFlowEnv()
        
        # Do some actions
        env.reset(task_id="task_easy_delivery")
        env.step(Action(tool_name="READ_TICKET"))
        env.step(Action(tool_name="GET_ORDER_DETAILS"))
        
        # Reset again
        env.reset(task_id="task_easy_delivery")
        state = env.state()
        
        assert state.step_count == 0
        assert state.ticket_read == False
        assert state.order_retrieved is None
        assert len(state.action_history) == 0
    
    def test_reset_loads_task(self):
        """Reset should load the specified task."""
        env = OpsFlowEnv()
        obs = env.reset(task_id="task_medium_refund")
        assert obs.task_id == "task_medium_refund"
    
    def test_reset_with_invalid_task_raises(self):
        """Reset with invalid task_id should raise ValueError."""
        env = OpsFlowEnv()
        with pytest.raises(ValueError):
            env.reset(task_id="nonexistent_task")
    
    def test_reset_loads_default_task(self):
        """Reset without task_id should load the first task."""
        env = OpsFlowEnv()
        obs = env.reset()
        assert obs.task_id == list_tasks()[0]


class TestEnvironmentStep:
    """Tests for environment step functionality."""
    
    def test_step_returns_step_result(self):
        """Step should return observation, reward, done, info."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        
        result = env.step(Action(tool_name="READ_TICKET"))
        
        assert isinstance(result.observation, Observation)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)
    
    def test_step_increments_counter(self):
        """Step should increment step counter."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        
        assert env.step_count == 0
        env.step(Action(tool_name="READ_TICKET"))
        assert env.step_count == 1
        env.step(Action(tool_name="GET_ORDER_DETAILS"))
        assert env.step_count == 2
    
    def test_step_records_history(self):
        """Step should record action in history."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        
        env.step(Action(tool_name="READ_TICKET", reasoning="Testing"))
        
        state = env.state()
        assert len(state.action_history) == 1
        assert state.action_history[0]["tool_name"] == "READ_TICKET"
    
    def test_step_updates_ticket_read(self):
        """READ_TICKET should update ticket_read flag."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        
        assert env.ticket_read == False
        env.step(Action(tool_name="READ_TICKET"))
        assert env.ticket_read == True
    
    def test_step_episode_ends_on_submit(self):
        """SUBMIT_RESOLUTION should end the episode."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        
        env.step(Action(tool_name="READ_TICKET"))
        result = env.step(Action(
            tool_name="SUBMIT_RESOLUTION",
            arguments={"status": "resolved"}
        ))
        
        assert result.done == True
    
    def test_step_episode_ends_on_max_steps(self):
        """Episode should end when max steps reached."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        
        # Do max steps without submitting
        for i in range(env.MAX_STEPS):
            result = env.step(Action(tool_name="READ_TICKET"))
        
        assert result.done == True
    
    def test_step_on_finished_episode_returns_zero_reward(self):
        """Step on finished episode should return 0 reward."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        
        env.step(Action(tool_name="SUBMIT_RESOLUTION"))
        result = env.step(Action(tool_name="READ_TICKET"))
        
        assert result.reward == 0.0
        assert result.done == True


class TestEnvironmentState:
    """Tests for environment state functionality."""
    
    def test_state_returns_environment_state(self):
        """State should return EnvironmentState object."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        
        state = env.state()
        assert isinstance(state, EnvironmentState)
    
    def test_state_contains_task_info(self):
        """State should contain task information."""
        env = OpsFlowEnv()
        env.reset(task_id="task_medium_refund")
        
        state = env.state()
        assert state.task_id == "task_medium_refund"
        assert state.task_difficulty == "medium"
    
    def test_state_contains_workflow_state(self):
        """State should contain workflow flags."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        env.step(Action(tool_name="READ_TICKET"))
        env.step(Action(tool_name="GET_ORDER_DETAILS"))
        
        state = env.state()
        assert state.ticket_read == True
        assert state.order_retrieved is not None


class TestToolExecution:
    """Tests for individual tool execution."""
    
    def test_read_ticket_returns_ticket_info(self):
        """READ_TICKET should return ticket information."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        
        result = env.step(Action(tool_name="READ_TICKET"))
        output = result.info["tool_output"]
        
        assert output["success"] == True
        assert "ticket_text" in output
        assert "order_id" in output
    
    def test_get_order_details_returns_order(self):
        """GET_ORDER_DETAILS should return order information."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        env.step(Action(tool_name="READ_TICKET"))
        
        result = env.step(Action(tool_name="GET_ORDER_DETAILS"))
        output = result.info["tool_output"]
        
        assert output["success"] == True
        assert "order" in output
    
    def test_check_policy_returns_policy(self):
        """CHECK_POLICY should return policy information."""
        env = OpsFlowEnv()
        env.reset(task_id="task_medium_refund")
        env.step(Action(tool_name="READ_TICKET"))
        env.step(Action(tool_name="GET_ORDER_DETAILS"))
        
        result = env.step(Action(tool_name="CHECK_POLICY"))
        output = result.info["tool_output"]
        
        assert output["success"] == True
        assert "policy" in output
        assert "approval_required" in output
    
    def test_execute_refund_completes(self):
        """EXECUTE_REFUND should process refund."""
        env = OpsFlowEnv()
        env.reset(task_id="task_medium_refund")
        env.step(Action(tool_name="READ_TICKET"))
        env.step(Action(tool_name="GET_ORDER_DETAILS"))
        env.step(Action(tool_name="CHECK_POLICY"))
        
        result = env.step(Action(tool_name="EXECUTE_REFUND"))
        output = result.info["tool_output"]
        
        assert output["success"] == True
        assert env.refund_executed == True
    
    def test_send_reply_requires_message(self):
        """SEND_CUSTOMER_REPLY should require a meaningful message."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        
        result = env.step(Action(
            tool_name="SEND_CUSTOMER_REPLY",
            arguments={"message": "hi"}  # Too short
        ))
        output = result.info["tool_output"]
        
        assert output["success"] == False


class TestRewardCalculation:
    """Tests for reward calculation."""
    
    def test_positive_reward_for_correct_action(self):
        """Correct actions should give positive reward."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        
        result = env.step(Action(tool_name="READ_TICKET"))
        assert result.reward > 0
    
    def test_penalty_for_duplicate_action(self):
        """Duplicate actions should be penalized."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        
        env.step(Action(tool_name="READ_TICKET"))
        result = env.step(Action(tool_name="READ_TICKET"))
        
        assert result.reward < 0
    
    def test_reward_breakdown_in_info(self):
        """Step info should contain reward breakdown."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        
        result = env.step(Action(tool_name="READ_TICKET"))
        
        assert "reward_breakdown" in result.info
        assert "total_reward" in result.info


class TestComplianceViolations:
    """Tests for compliance violation detection."""
    
    def test_order_lookup_before_ticket_read_flagged(self):
        """Looking up order before reading ticket should be flagged."""
        env = OpsFlowEnv()
        env.reset(task_id="task_easy_delivery")
        
        # Skip reading ticket
        env.step(Action(tool_name="GET_ORDER_DETAILS"))
        
        state = env.state()
        assert "order_lookup_before_ticket_read" in state.compliance_violations
    
    def test_approval_bypass_flagged(self):
        """Bypassing approval for high-value refund should be flagged."""
        env = OpsFlowEnv()
        env.reset(task_id="task_hard_approval")
        
        env.step(Action(tool_name="READ_TICKET"))
        env.step(Action(tool_name="GET_ORDER_DETAILS"))
        # Skip approval, go straight to refund
        env.step(Action(tool_name="EXECUTE_REFUND"))
        
        state = env.state()
        assert "refund_without_required_approval" in state.compliance_violations
