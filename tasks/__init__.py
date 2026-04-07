"""
OpenEnv-OpsFlow: Task Definitions

Defines the three required tasks (easy, medium, hard) for the environment.
Each task includes ticket text, expected workflow, and grading criteria.
"""

from typing import Dict, Any, List
from models import TaskDefinition


# =============================================================================
# Task 1: Delivery Status Resolution (EASY)
# =============================================================================

TASK_EASY_DELIVERY = TaskDefinition(
    task_id="task_easy_delivery",
    difficulty="easy",
    ticket_text="""Subject: Where is my order?

Hi Support,

I ordered some Wireless Bluetooth Headphones on March 15th (Order #ORD-001) 
and I haven't received them yet. Can you tell me where my order is?

Thanks,
Alice""",
    description="Customer asking about delivery status. Simple lookup and response required.",
    order_id="ORD-001",
    customer_id="CUST-001",
    expected_workflow=[
        "READ_TICKET",
        "GET_ORDER_DETAILS",
        "SEND_CUSTOMER_REPLY",
        "SUBMIT_RESOLUTION"
    ],
    requires_approval=False,
    requires_policy_check=False,
    success_criteria={
        "order_retrieved": True,
        "customer_reply_sent": True,
        "resolution_submitted": True,
        "reply_contains_status": True
    }
)


# =============================================================================
# Task 2: Low-Value Refund with Policy Check (MEDIUM)
# =============================================================================

TASK_MEDIUM_REFUND = TaskDefinition(
    task_id="task_medium_refund",
    difficulty="medium",
    ticket_text="""Subject: Refund request - Damaged Coffee Maker

Hello,

I received my Premium Coffee Maker (Order #ORD-002) on March 14th, 
but unfortunately it arrived with a cracked water reservoir. 
The product was $89.99 and I would like a full refund please.

I have photos of the damage if needed.

Best regards,
Bob Smith""",
    description="Customer requesting refund for damaged item under threshold. Requires policy check before refund.",
    order_id="ORD-002",
    customer_id="CUST-002",
    expected_workflow=[
        "READ_TICKET",
        "GET_ORDER_DETAILS",
        "CHECK_POLICY",
        "EXECUTE_REFUND",
        "SEND_CUSTOMER_REPLY",
        "SUBMIT_RESOLUTION"
    ],
    requires_approval=False,
    requires_policy_check=True,
    success_criteria={
        "order_retrieved": True,
        "policy_checked": True,
        "refund_executed": True,
        "customer_reply_sent": True,
        "resolution_submitted": True
    }
)


# =============================================================================
# Task 3: High-Value Refund Requiring Approval (HARD)
# =============================================================================

TASK_HARD_APPROVAL = TaskDefinition(
    task_id="task_hard_approval",
    difficulty="hard",
    ticket_text="""Subject: URGENT - Defective TV, need refund

Dear Support Team,

I am extremely disappointed. I purchased a 4K Smart Television 55-inch 
(Order #ORD-003) for $599.99 and after just 3 days of use, the screen 
has developed multiple dead pixels and a dark band across the middle.

This is completely unacceptable for a premium product. I demand a full 
refund immediately.

I am a long-time VIP customer and expect this to be handled properly.

Carol Williams""",
    description="VIP customer requesting high-value refund for defective product. Requires policy check AND manager approval due to amount.",
    order_id="ORD-003",
    customer_id="CUST-003",
    expected_workflow=[
        "READ_TICKET",
        "GET_ORDER_DETAILS",
        "GET_CUSTOMER_PROFILE",
        "CHECK_POLICY",
        "REQUEST_APPROVAL",
        "EXECUTE_REFUND",
        "SEND_CUSTOMER_REPLY",
        "SUBMIT_RESOLUTION"
    ],
    requires_approval=True,
    requires_policy_check=True,
    success_criteria={
        "order_retrieved": True,
        "customer_retrieved": True,
        "policy_checked": True,
        "approval_requested": True,
        "approval_status": "approved",
        "refund_executed": True,
        "customer_reply_sent": True,
        "resolution_submitted": True
    }
)


# =============================================================================
# Task Registry
# =============================================================================

TASKS: Dict[str, TaskDefinition] = {
    "task_easy_delivery": TASK_EASY_DELIVERY,
    "task_medium_refund": TASK_MEDIUM_REFUND,
    "task_hard_approval": TASK_HARD_APPROVAL,
}


def get_task(task_id: str) -> TaskDefinition:
    """Get a task by ID."""
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(TASKS.keys())}")
    return TASKS[task_id]


def list_tasks() -> List[str]:
    """List all available task IDs."""
    return list(TASKS.keys())


def get_all_tasks() -> Dict[str, TaskDefinition]:
    """Get all tasks."""
    return TASKS.copy()
