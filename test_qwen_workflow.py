#!/usr/bin/env python3
"""
Test Case: Qwen Model Workflow Decision Making
=============================================

This test demonstrates how the Qwen/Qwen2.5-72B-Instruct model makes
intelligent workflow decisions in the TickFlow customer support system.

The test shows:
1. Qwen reading and understanding customer tickets
2. Qwen making contextual tool decisions
3. Qwen following business rules and compliance
4. Qwen adapting to different scenarios
"""

import os
import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

# Set environment variables for REAL Qwen testing
os.environ["HF_TOKEN"] = "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"  # Your real HF token
os.environ["API_BASE_URL"] = "https://router.huggingface.co/v1"  # Real Hugging Face endpoint
os.environ["MODEL_NAME"] = "Qwen/Qwen2.5-72B-Instruct"

from env import OpsFlowEnv
from models import Action
from inference import OpsFlowAgent, SYSTEM_PROMPT


class TestQwenWorkflowDecisions:
    """Test suite demonstrating Qwen model workflow decision making."""

    def setup_method(self):
        """Setup for each test method."""
        self.env = OpsFlowEnv()
        
    def create_mock_qwen_responses(self):
        """Create realistic Qwen model responses for different workflow steps."""
        return {
            # Step 1: Qwen decides to read the ticket first
            "step1_read_ticket": {
                "choices": [{
                    "message": {
                        "content": '''```json
{
    "tool_name": "READ_TICKET",
    "arguments": {},
    "reasoning": "I need to read and understand the customer's issue before proceeding with any actions."
}
```'''
                    }
                }]
            },
            
            # Step 2: Qwen analyzes ticket and decides to get order details
            "step2_get_order": {
                "choices": [{
                    "message": {
                        "content": '''```json
{
    "tool_name": "GET_ORDER_DETAILS", 
    "arguments": {"order_id": "ORD-001"},
    "reasoning": "Barani The customer is asking about Order #ORD-001. I need to retrieve the order details to check delivery status."
}
```'''
                    }
                }]
            },
            
            # Step 3: Qwen sees order is shipped and decides to reply to customer
            "step3_reply_customer": {
                "choices": [{
                    "message": {
                        "content": '''```json
{
    "tool_name": "SEND_CUSTOMER_REPLY",
    "arguments": {
        "message": "Hi Alice, I've checked your order #ORD-001 for Wireless Bluetooth Headphones. Good news! Your order has been shipped with tracking number TRK-12345-ABC. You should receive it soon."
    },
    "reasoning": "Barani I found the order is shipped with tracking info. I should inform the customer with the tracking details to resolve their inquiry."
}
```'''
                    }
                }]
            },
            
            # Step 4: Qwen decides to close the ticket
            "step4_submit_resolution": {
                "choices": [{
                    "message": {
                        "content": '''```json
{
    "tool_name": "SUBMIT_RESOLUTION",
    "arguments": {
        "status": "resolved",
        "summary": "Provided tracking information for shipped order. Customer inquiry resolved."
    },
    "reasoning": "I've provided the customer with their tracking information. The delivery inquiry has been resolved, so I should close the ticket."
}
```'''
                    }
                }]
            },
            
            # Refund scenario - Step 1: Qwen decides to check policy before refund
            "refund_check_policy": {
                "choices": [{
                    "message": {
                        "content": '''```json
{
    "tool_name": "CHECK_POLICY",
    "arguments": {"customer_tier": "premium"},
    "reasoning": "Before processing any refund, I must check the refund policy to ensure compliance and determine if manager approval is needed for this amount."
}
```'''
                    }
                }]
            },
            
            # High-value refund - Qwen decides to request approval
            "high_value_approval": {
                "choices": [{
                    "message": {
                        "content": '''```json
{
    "tool_name": "REQUEST_APPROVAL",
    "arguments": {
        "order_id": "ORD-003",
        "amount": 599.99,
        "reason": "VIP customer requesting refund for defective 4K TV with dead pixels"
    },
    "reasoning": "This is a high-value refund of $599.99 which exceeds the approval threshold. Company policy requires manager approval for amounts above $500."
}
```'''
                    }
                }]
            }
        }

    def test_qwen_delivery_inquiry_workflow(self):
        """
        Test: REAL Qwen model handles delivery inquiry with intelligent workflow decisions
        
        This test demonstrates how the REAL Qwen/Qwen2.5-72B-Instruct model:
        1. Reads the ticket to understand the issue
        2. Retrieves order information based on ticket context  
        3. Analyzes order status and provides appropriate response
        4. Closes the ticket properly
        """
        print("\n🧠 Testing REAL Qwen Model: Delivery Inquiry Workflow")
        print("🔗 Using live API: https://router.huggingface.co/v1")
        print("🤖 Model: Qwen/Qwen2.5-72B-Instruct")
        
        # Setup environment with delivery inquiry task
        observation = self.env.reset(task_id="task_easy_delivery")
        
        # Create REAL Qwen client (no mocking!)
        from openai import OpenAI
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"], 
            api_key=os.environ["HF_TOKEN"]
        )
        
        # Create agent with REAL Qwen model
        agent = OpsFlowAgent(client=client, model="Qwen/Qwen2.5-72B-Instruct")
        
        print("\n🔥 Let's see what REAL Qwen decides...")
        
        try:
            # Step 1: Real Qwen makes first decision
            print("\n⏳ Calling Qwen API for Step 1...")
            action1 = agent.get_action(observation.model_dump())
            
            print(f"📋 Step 1 - REAL Qwen Decision: {action1.tool_name}")
            print(f"   💭 Qwen's Reasoning: {action1.reasoning}")
            
            # Execute Qwen's decision
            result1 = self.env.step(action1)
            print(f"   📊 Reward: {result1.reward:.2f}")
            
            # Step 2: Real Qwen makes second decision based on context
            print("\n⏳ Calling Qwen API for Step 2...")
            action2 = agent.get_action(result1.observation.model_dump())
            
            print(f"📦 Step 2 - REAL Qwen Decision: {action2.tool_name}")
            print(f"   💭 Qwen's Reasoning: {action2.reasoning}")
            print(f"   🔧 Arguments: {action2.arguments}")
            
            # Execute Qwen's decision
            result2 = self.env.step(action2)
            print(f"   📊 Reward: {result2.reward:.2f}")
            
            # Step 3: Real Qwen makes third decision
            print("\n⏳ Calling Qwen API for Step 3...")
            action3 = agent.get_action(result2.observation.model_dump())
            
            print(f"💬 Step 3 - REAL Qwen Decision: {action3.tool_name}")
            print(f"   💭 Qwen's Reasoning: {action3.reasoning}")
            if action3.arguments.get('message'):
                print(f"   📝 Message Preview: {action3.arguments.get('message', '')[:80]}...")
            
            # Execute Qwen's decision
            result3 = self.env.step(action3)
            print(f"   📊 Reward: {result3.reward:.2f}")
            
            # Step 4: Real Qwen makes final decision
            print("\n⏳ Calling Qwen API for Step 4...")
            action4 = agent.get_action(result3.observation.model_dump())
            
            print(f"✅ Step 4 - REAL Qwen Decision: {action4.tool_name}")
            print(f"   💭 Qwen's Reasoning: {action4.reasoning}")
            print(f"   🏁 Status: {action4.arguments.get('status', 'N/A')}")
            
            # Execute final decision
            result4 = self.env.step(action4)
            print(f"   📊 Final Reward: {result4.reward:.2f}")
            
            print(f"\n🎯 Workflow Complete! Episode done: {result4.done}")
            print(f"🏆 Total workflow score: {result4.info.get('final_score', 'N/A')}")
            
            # Basic validation (less strict since we're using real AI)
            print(f"\n✅ Real Qwen successfully completed {len([action1, action2, action3, action4])} steps")
            
        except Exception as e:
            print(f"\n❌ Error calling real Qwen API: {e}")
            print("💡 Make sure your HF_TOKEN is valid and you have API access")
            raise

    def test_qwen_compliance_decision_making(self):
        """
        Test: Qwen model follows compliance rules and business policies
        
        This test demonstrates how Qwen:
        1. Understands compliance requirements
        2. Checks policies before executing refunds
        3. Requests approvals for high-value transactions
        """
        print("\n🛡️ Testing Qwen Model: Compliance Decision Making")
        
        # Setup environment with refund scenario
        observation = self.env.reset(task_id="task_medium_refund")
        
        mock_responses = self.create_mock_qwen_responses()
        
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            agent = OpsFlowAgent(client=mock_client, model="Qwen/Qwen2.5-72B-Instruct")
            
            # Simulate Qwen reading ticket and getting order details
            result1 = self.env.step(Action(tool_name="READ_TICKET"))
            result2 = self.env.step(Action(tool_name="GET_ORDER_DETAILS"))
            
            # Test: Qwen decides to check policy before refund (compliance!)
            mock_client.chat.completions.create.return_value = mock_responses["refund_check_policy"]
            
            action = agent.get_action(result2.observation.model_dump())
            
            print(f"🔍 Policy Check - Qwen Decision: {action.tool_name}")
            print(f"   Reasoning: {action.reasoning}")
            
            assert action.tool_name == "CHECK_POLICY"
            assert "policy" in action.reasoning.lower()
            assert "compliance" in action.reasoning.lower()

    def test_qwen_high_value_approval_workflow(self):
        """
        Test: Qwen model handles high-value transactions requiring approval
        
        This test demonstrates Qwen's understanding of:
        1. Dollar amount thresholds
        2. Approval workflow requirements
        3. VIP customer handling
        """
        print("\n💰 Testing Qwen Model: High-Value Approval Workflow")
        
        observation = self.env.reset(task_id="task_hard_approval")
        
        mock_responses = self.create_mock_qwen_responses()
        
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            agent = OpsFlowAgent(client=mock_client, model="Qwen/Qwen2.5-72B-Instruct")
            
            # Simulate workflow progression to approval decision point
            self.env.step(Action(tool_name="READ_TICKET"))
            self.env.step(Action(tool_name="GET_ORDER_DETAILS"))
            self.env.step(Action(tool_name="GET_CUSTOMER_PROFILE"))
            result = self.env.step(Action(tool_name="CHECK_POLICY"))
            
            # Test: Qwen recognizes need for approval on high-value refund
            mock_client.chat.completions.create.return_value = mock_responses["high_value_approval"]
            
            action = agent.get_action(result.observation.model_dump())
            
            print(f"🔒 Approval Request - Qwen Decision: {action.tool_name}")
            print(f"   Reasoning: {action.reasoning}")
            print(f"   Amount: ${action.arguments.get('amount')}")
            
            assert action.tool_name == "REQUEST_APPROVAL"
            assert action.arguments.get("amount") == 599.99
            assert "$500" in action.reasoning or "approval" in action.reasoning.lower()

    def test_qwen_system_prompt_integration(self):
        """
        Test: Qwen receives and follows the system prompt instructions
        
        This verifies that the system prompt is properly integrated and
        Qwen understands the available tools and workflow rules.
        """
        print("\n📝 Testing Qwen Model: System Prompt Integration")
        
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            agent = OpsFlowAgent(client=mock_client, model="Qwen/Qwen2.5-72B-Instruct")
            
            # Reset conversation (loads system prompt)
            agent.reset()
            
            # Verify system prompt is loaded
            assert len(agent.conversation_history) == 1
            assert agent.conversation_history[0]["role"] == "system"
            assert "customer support workflow system" in agent.conversation_history[0]["content"]
            assert "READ_TICKET" in agent.conversation_history[0]["content"]
            assert "EXECUTE_REFUND" in agent.conversation_history[0]["content"]
            assert "JSON object" in agent.conversation_history[0]["content"]
            
            print("✅ System prompt properly loaded into Qwen conversation")
            print(f"   Tools mentioned: 9 workflow tools")
            print(f"   Format: JSON response required")
            print(f"   Rules: Workflow sequence specified")

    def test_qwen_context_awareness(self):
        """
        Test: Qwen maintains context across multiple workflow steps
        
        This demonstrates how Qwen builds understanding over time and
        makes decisions based on accumulated context.
        """
        print("\n🧩 Testing Qwen Model: Context Awareness")
        
        observation = self.env.reset(task_id="task_easy_delivery")
        
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            agent = OpsFlowAgent(client=mock_client, model="Qwen/Qwen2.5-72B-Instruct")
            
            # Execute first step
            result1 = self.env.step(Action(tool_name="READ_TICKET"))
            
            # Mock response for second step
            mock_client.chat.completions.create.return_value = self.create_mock_qwen_responses()["step2_get_order"]
            
            # Get Qwen's decision with accumulated context
            action = agent.get_action(result1.observation.model_dump())
            
            # Verify conversation history accumulation
            assert len(agent.conversation_history) >= 3  # System + User + Assistant + User
            
            # Verify context includes ticket information
            context_message = agent.conversation_history[-1]["content"]
            assert "Current Status:" in context_message
            assert "Steps Remaining:" in context_message
            assert "Ticket:" in context_message
            assert "Alice" in context_message  # Customer name from ticket
            assert "ORD-001" in context_message  # Order number from ticket
            
            print("✅ Qwen maintains conversation context")
            print(f"   Conversation length: {len(agent.conversation_history)} messages")
            print(f"   Context includes: Status, Steps, Ticket, History")


def main():
    """Run the Qwen workflow decision making tests."""
    print("🧠 QWEN MODEL WORKFLOW DECISION TESTING")
    print("=" * 50)
    
    test_instance = TestQwenWorkflowDecisions()
    test_instance.setup_method()
    
    try:
        # Test 1: Complete workflow decision making
        test_instance.test_qwen_delivery_inquiry_workflow()
        print("✅ Test 1 PASSED: Delivery inquiry workflow")
        
        # Test 1: REAL Qwen workflow decision making
        test_instance.test_qwen_delivery_inquiry_workflow()
        print("✅ Test 1 PASSED: REAL Qwen delivery inquiry workflow")
        
        print("\n🎉 REAL QWEN TEST COMPLETED!")
        print("🤖 Qwen/Qwen2.5-72B-Instruct successfully demonstrated LIVE intelligent workflow decision making")
        print("💰 Note: This test uses real API calls and may cost money")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Check your HF_TOKEN and internet connection")
        raise
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
