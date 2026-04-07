"""
OpenEnv-OpsFlow: Test Suite for API
"""

import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_root(self, client):
        """Root endpoint should return health status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_endpoint(self, client):
        """Health endpoint should return health status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestResetEndpoint:
    """Tests for reset endpoint."""
    
    def test_reset_default_task(self, client):
        """Reset without task_id should load default task."""
        response = client.post("/reset", json={})
        assert response.status_code == 200
        data = response.json()
        assert "observation" in data
        assert "task_id" in data
    
    def test_reset_specific_task(self, client):
        """Reset with task_id should load that task."""
        response = client.post("/reset", json={"task_id": "task_medium_refund"})
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task_medium_refund"
    
    def test_reset_invalid_task(self, client):
        """Reset with invalid task_id should return 400."""
        response = client.post("/reset", json={"task_id": "invalid_task"})
        assert response.status_code == 400
    
    def test_reset_returns_observation(self, client):
        """Reset should return full observation."""
        response = client.post("/reset", json={"task_id": "task_easy_delivery"})
        assert response.status_code == 200
        data = response.json()
        
        obs = data["observation"]
        assert "task_id" in obs
        assert "ticket_text" in obs
        assert "available_tools" in obs


class TestStepEndpoint:
    """Tests for step endpoint."""
    
    def test_step_requires_reset(self, client):
        """Step without reset should still work (auto-initializes)."""
        # First reset to ensure clean state
        client.post("/reset", json={})
        
        action = {
            "action": {
                "tool_name": "READ_TICKET",
                "arguments": {}
            }
        }
        response = client.post("/step", json=action)
        assert response.status_code == 200
    
    def test_step_returns_result(self, client):
        """Step should return observation, reward, done, info."""
        client.post("/reset", json={"task_id": "task_easy_delivery"})
        
        action = {
            "action": {
                "tool_name": "READ_TICKET",
                "arguments": {},
                "reasoning": "Starting workflow"
            }
        }
        response = client.post("/step", json=action)
        assert response.status_code == 200
        
        data = response.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data
    
    def test_step_updates_observation(self, client):
        """Step should update the observation."""
        client.post("/reset", json={"task_id": "task_easy_delivery"})
        
        # Take a step
        action = {"action": {"tool_name": "READ_TICKET", "arguments": {}}}
        response = client.post("/step", json=action)
        data = response.json()
        
        # Check observation was updated
        obs = data["observation"]
        assert obs["current_status"] == "ticket_read"
        assert len(obs["workflow_history"]) == 1
    
    def test_step_done_on_submit(self, client):
        """SUBMIT_RESOLUTION should set done=True."""
        client.post("/reset", json={"task_id": "task_easy_delivery"})
        
        action = {"action": {"tool_name": "SUBMIT_RESOLUTION", "arguments": {"status": "resolved"}}}
        response = client.post("/step", json=action)
        data = response.json()
        
        assert data["done"] == True


class TestStateEndpoint:
    """Tests for state endpoint."""
    
    def test_state_after_reset(self, client):
        """State should return environment state after reset."""
        client.post("/reset", json={"task_id": "task_easy_delivery"})
        
        response = client.get("/state")
        assert response.status_code == 200
        
        data = response.json()
        assert "state" in data
        state = data["state"]
        assert state["task_id"] == "task_easy_delivery"
    
    def test_state_reflects_actions(self, client):
        """State should reflect actions taken."""
        client.post("/reset", json={"task_id": "task_easy_delivery"})
        
        # Take some steps
        client.post("/step", json={"action": {"tool_name": "READ_TICKET", "arguments": {}}})
        client.post("/step", json={"action": {"tool_name": "GET_ORDER_DETAILS", "arguments": {}}})
        
        response = client.get("/state")
        data = response.json()
        state = data["state"]
        
        assert state["ticket_read"] == True
        assert state["order_retrieved"] is not None
        assert state["step_count"] == 2


class TestTasksEndpoint:
    """Tests for tasks endpoint."""
    
    def test_list_tasks(self, client):
        """Tasks endpoint should list all tasks."""
        response = client.get("/tasks")
        assert response.status_code == 200
        
        data = response.json()
        assert "tasks" in data
        assert len(data["tasks"]) >= 3
    
    def test_task_info(self, client):
        """Tasks should have proper info."""
        response = client.get("/tasks")
        data = response.json()
        
        for task in data["tasks"]:
            assert "task_id" in task
            assert "difficulty" in task
            assert "description" in task
    
    def test_get_specific_task(self, client):
        """Should be able to get specific task info."""
        response = client.get("/tasks/task_easy_delivery")
        assert response.status_code == 200
        
        data = response.json()
        assert data["task_id"] == "task_easy_delivery"
        assert data["difficulty"] == "easy"
    
    def test_get_invalid_task(self, client):
        """Invalid task should return 404."""
        response = client.get("/tasks/invalid_task")
        assert response.status_code == 404


class TestFullWorkflow:
    """Integration tests for complete workflows."""
    
    def test_easy_task_complete_workflow(self, client):
        """Complete easy task workflow."""
        # Reset
        client.post("/reset", json={"task_id": "task_easy_delivery"})
        
        # Execute workflow
        steps = [
            {"tool_name": "READ_TICKET", "arguments": {}},
            {"tool_name": "GET_ORDER_DETAILS", "arguments": {}},
            {"tool_name": "SEND_CUSTOMER_REPLY", "arguments": {"message": "Your order has been shipped and is on the way. Tracking: TRK-12345-ABC"}},
            {"tool_name": "SUBMIT_RESOLUTION", "arguments": {"status": "resolved", "summary": "Provided tracking info"}}
        ]
        
        for step in steps:
            response = client.post("/step", json={"action": step})
            assert response.status_code == 200
        
        # Check final state
        response = client.get("/state")
        state = response.json()["state"]
        
        assert state["done"] == True
        assert state["resolution_submitted"] == True
    
    def test_medium_task_workflow(self, client):
        """Complete medium task workflow."""
        client.post("/reset", json={"task_id": "task_medium_refund"})
        
        steps = [
            {"tool_name": "READ_TICKET", "arguments": {}},
            {"tool_name": "GET_ORDER_DETAILS", "arguments": {}},
            {"tool_name": "CHECK_POLICY", "arguments": {}},
            {"tool_name": "EXECUTE_REFUND", "arguments": {}},
            {"tool_name": "SEND_CUSTOMER_REPLY", "arguments": {"message": "Your refund of $89.99 has been processed and will appear in 3-5 business days."}},
            {"tool_name": "SUBMIT_RESOLUTION", "arguments": {"status": "resolved"}}
        ]
        
        for step in steps:
            response = client.post("/step", json={"action": step})
            assert response.status_code == 200
        
        # Verify state
        response = client.get("/state")
        state = response.json()["state"]
        
        assert state["refund_executed"] == True
        assert state["policy_checked"] == True
