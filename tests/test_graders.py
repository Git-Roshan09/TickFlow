"""
OpenEnv-OpsFlow: Test Suite for Graders
"""

import pytest
from graders import grade_task, DeliveryStatusGrader, LowValueRefundGrader, HighValueApprovalGrader
from models import EnvironmentState


class TestDeliveryStatusGrader:
    """Tests for Task 1: Delivery Status grader."""
    
    def create_state(self, **kwargs) -> EnvironmentState:
        """Create a test state with default values."""
        defaults = {
            "task_id": "task_easy_delivery",
            "task_difficulty": "easy",
            "ticket_text": "Where is my order?",
            "step_count": 4,
            "max_steps": 15,
            "done": True,
            "action_history": [],
            "compliance_violations": [],
            "total_reward": 0.5
        }
        defaults.update(kwargs)
        return EnvironmentState(**defaults)
    
    def test_perfect_score(self):
        """Perfect workflow should score 1.0."""
        state = self.create_state(
            order_retrieved={"order_id": "ORD-001"},
            customer_reply_sent=True,
            customer_reply_content="Your order TRK-12345 has been shipped and is on the way.",
            resolution_submitted=True
        )
        
        grader = DeliveryStatusGrader()
        score = grader.grade(state)
        
        assert score == 1.0
    
    def test_partial_score_no_reply(self):
        """Missing customer reply should reduce score."""
        state = self.create_state(
            order_retrieved={"order_id": "ORD-001"},
            customer_reply_sent=False,
            resolution_submitted=True
        )
        
        grader = DeliveryStatusGrader()
        score = grader.grade(state)
        
        assert 0.3 <= score < 0.7
    
    def test_zero_score_no_order(self):
        """No order retrieved should score low."""
        state = self.create_state(
            order_retrieved=None,
            customer_reply_sent=False,
            resolution_submitted=True
        )
        
        grader = DeliveryStatusGrader()
        score = grader.grade(state)
        
        assert score <= 0.3
    
    def test_compliance_violation_penalty(self):
        """Compliance violations should reduce score."""
        state = self.create_state(
            order_retrieved={"order_id": "ORD-001"},
            customer_reply_sent=True,
            customer_reply_content="Your order has been shipped.",
            resolution_submitted=True,
            compliance_violations=["order_lookup_before_ticket_read"]
        )
        
        grader = DeliveryStatusGrader()
        score = grader.grade(state)
        
        assert score < 1.0


class TestLowValueRefundGrader:
    """Tests for Task 2: Low-Value Refund grader."""
    
    def create_state(self, **kwargs) -> EnvironmentState:
        """Create a test state with default values."""
        defaults = {
            "task_id": "task_medium_refund",
            "task_difficulty": "medium",
            "ticket_text": "I want a refund for my damaged coffee maker.",
            "step_count": 6,
            "max_steps": 15,
            "done": True,
            "action_history": [],
            "compliance_violations": [],
            "total_reward": 0.5
        }
        defaults.update(kwargs)
        return EnvironmentState(**defaults)
    
    def test_perfect_score(self):
        """Perfect workflow should score 1.0."""
        state = self.create_state(
            order_retrieved={"order_id": "ORD-002", "order_amount": 89.99},
            policy_checked=True,
            refund_executed=True,
            refund_amount=89.99,
            customer_reply_sent=True,
            customer_reply_content="Your refund of $89.99 has been processed.",
            resolution_submitted=True
        )
        
        grader = LowValueRefundGrader()
        score = grader.grade(state)
        
        assert score == 1.0
    
    def test_refund_without_policy_check(self):
        """Refund without policy check should score lower."""
        state = self.create_state(
            order_retrieved={"order_id": "ORD-002", "order_amount": 89.99},
            policy_checked=False,  # Skipped policy check
            refund_executed=True,
            customer_reply_sent=True,
            customer_reply_content="Your refund has been processed.",
            resolution_submitted=True
        )
        
        grader = LowValueRefundGrader()
        score = grader.grade(state)
        
        assert 0.4 <= score <= 0.6
    
    def test_no_refund_executed(self):
        """No refund executed should score poorly."""
        state = self.create_state(
            order_retrieved={"order_id": "ORD-002"},
            policy_checked=True,
            refund_executed=False,  # No refund
            customer_reply_sent=True,
            resolution_submitted=True
        )
        
        grader = LowValueRefundGrader()
        score = grader.grade(state)
        
        assert score < 0.6


class TestHighValueApprovalGrader:
    """Tests for Task 3: High-Value Approval grader."""
    
    def create_state(self, **kwargs) -> EnvironmentState:
        """Create a test state with default values."""
        defaults = {
            "task_id": "task_hard_approval",
            "task_difficulty": "hard",
            "ticket_text": "I need a refund for my defective TV.",
            "step_count": 8,
            "max_steps": 15,
            "done": True,
            "action_history": [],
            "compliance_violations": [],
            "total_reward": 0.5
        }
        defaults.update(kwargs)
        return EnvironmentState(**defaults)
    
    def test_perfect_score(self):
        """Perfect workflow should score 1.0."""
        state = self.create_state(
            order_retrieved={"order_id": "ORD-003", "order_amount": 599.99},
            customer_retrieved={"customer_id": "CUST-003", "tier": "vip"},
            policy_checked=True,
            approval_requested=True,
            approval_status="approved",
            refund_executed=True,
            refund_amount=599.99,
            customer_reply_sent=True,
            customer_reply_content="Your refund has been approved and processed.",
            resolution_submitted=True
        )
        
        grader = HighValueApprovalGrader()
        score = grader.grade(state)
        
        assert score == 1.0
    
    def test_approval_bypassed(self):
        """Bypassing approval should heavily penalize score."""
        state = self.create_state(
            order_retrieved={"order_id": "ORD-003", "order_amount": 599.99},
            customer_retrieved={"customer_id": "CUST-003", "tier": "vip"},
            policy_checked=True,
            approval_requested=False,  # No approval
            refund_executed=True,  # Refund anyway
            customer_reply_sent=True,
            resolution_submitted=True
        )
        
        grader = HighValueApprovalGrader()
        score = grader.grade(state)
        
        assert score < 0.3  # Heavy penalty
    
    def test_partial_workflow(self):
        """Partial workflow should score in middle range."""
        state = self.create_state(
            order_retrieved={"order_id": "ORD-003", "order_amount": 599.99},
            customer_retrieved={"customer_id": "CUST-003", "tier": "vip"},
            policy_checked=True,
            approval_requested=True,
            approval_status="approved",
            refund_executed=True,
            customer_reply_sent=False,  # Forgot reply
            resolution_submitted=True
        )
        
        grader = HighValueApprovalGrader()
        score = grader.grade(state)
        
        assert 0.5 <= score <= 0.9


class TestGradeTaskFunction:
    """Tests for the grade_task helper function."""
    
    def test_grade_task_easy(self):
        """grade_task should work for easy task."""
        state = EnvironmentState(
            task_id="task_easy_delivery",
            task_difficulty="easy",
            ticket_text="",
            step_count=4,
            max_steps=15,
            done=True,
            order_retrieved={"order_id": "ORD-001"},
            customer_reply_sent=True,
            customer_reply_content="Order shipped with tracking.",
            resolution_submitted=True
        )
        
        score = grade_task("task_easy_delivery", state)
        assert 0.0 <= score <= 1.0
    
    def test_grade_task_unknown_raises(self):
        """grade_task with unknown task should raise ValueError."""
        state = EnvironmentState(
            task_id="unknown",
            task_difficulty="easy",
            ticket_text="",
            step_count=0,
            max_steps=15,
            done=True
        )
        
        with pytest.raises(ValueError):
            grade_task("unknown_task", state)


class TestScoreRange:
    """Tests to ensure all scores are in valid range."""
    
    def test_delivery_score_range(self):
        """Delivery grader should always return 0.0-1.0."""
        grader = DeliveryStatusGrader()
        
        # Test various states
        states = [
            EnvironmentState(task_id="", task_difficulty="easy", ticket_text="", step_count=0, max_steps=15, done=True),
            EnvironmentState(task_id="", task_difficulty="easy", ticket_text="", step_count=0, max_steps=15, done=True,
                           order_retrieved={"id": "1"}, customer_reply_sent=True, customer_reply_content="test message here",
                           resolution_submitted=True),
            EnvironmentState(task_id="", task_difficulty="easy", ticket_text="", step_count=0, max_steps=15, done=True,
                           compliance_violations=["v1", "v2", "v3", "v4", "v5"])
        ]
        
        for state in states:
            score = grader.grade(state)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"
    
    def test_refund_score_range(self):
        """Refund grader should always return 0.0-1.0."""
        grader = LowValueRefundGrader()
        
        states = [
            EnvironmentState(task_id="", task_difficulty="medium", ticket_text="", step_count=0, max_steps=15, done=True),
            EnvironmentState(task_id="", task_difficulty="medium", ticket_text="", step_count=0, max_steps=15, done=True,
                           order_retrieved={"id": "1"}, policy_checked=True, refund_executed=True,
                           customer_reply_sent=True, customer_reply_content="refund processed",
                           resolution_submitted=True),
        ]
        
        for state in states:
            score = grader.grade(state)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"
    
    def test_approval_score_range(self):
        """Approval grader should always return 0.0-1.0."""
        grader = HighValueApprovalGrader()
        
        states = [
            EnvironmentState(task_id="", task_difficulty="hard", ticket_text="", step_count=0, max_steps=15, done=True),
            EnvironmentState(task_id="", task_difficulty="hard", ticket_text="", step_count=0, max_steps=15, done=True,
                           order_retrieved={"id": "1"}, customer_retrieved={"id": "1"}, policy_checked=True,
                           approval_requested=True, approval_status="approved", refund_executed=True,
                           customer_reply_sent=True, customer_reply_content="approved refund",
                           resolution_submitted=True),
            EnvironmentState(task_id="", task_difficulty="hard", ticket_text="", step_count=0, max_steps=15, done=True,
                           refund_executed=True, compliance_violations=["v1", "v2"])  # Bad case
        ]
        
        for state in states:
            score = grader.grade(state)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"
