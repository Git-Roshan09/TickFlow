# OpenEnv-OpsFlow

## Policy-Aware Customer Support & Compliance Orchestrator

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://github.com/openenv)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, real-world OpenEnv environment where AI agents learn to resolve enterprise customer support tickets safely, efficiently, and compliantly through realistic tool orchestration.

---

## 🎯 Environment Description & Motivation

Modern enterprises receive customer support requests that require more than just answering a question. Many tickets need the agent to:

1. Inspect order records
2. Check company refund policy
3. Verify customer eligibility
4. Request manager approval when needed
5. Apply the correct business action
6. Send a compliant final response

**This is not a chatbot project.** This is a **workflow decision and execution environment** that simulates how support teams, refund teams, and compliance teams work in real companies.

### Why This Environment?

- **Real-world task**: Models genuine enterprise support operations
- **Industrial relevance**: Companies already use agents for refund processing, escalation routing, and compliance workflows
- **Clear evaluation**: Easy to grade deterministically based on workflow correctness
- **Agent training value**: Tests planning, sequencing, policy compliance, and recovery from bad actions

---

## 🔧 Action Space

The agent interacts through 9 tools that simulate enterprise operations:

| Tool | Description | Arguments |
|------|-------------|-----------|
| `READ_TICKET` | Read the customer support ticket | None |
| `GET_ORDER_DETAILS` | Retrieve order information | `order_id` (optional) |
| `GET_CUSTOMER_PROFILE` | Get customer profile and tier | `customer_id` (optional) |
| `CHECK_POLICY` | Check applicable refund/return policy | `customer_tier` (optional) |
| `REQUEST_APPROVAL` | Request manager approval | `order_id`, `amount`, `reason` |
| `EXECUTE_REFUND` | Process a refund | `order_id`, `amount`, `reason` |
| `ISSUE_STORE_CREDIT` | Issue store credit | `customer_id`, `amount`, `reason` |
| `SEND_CUSTOMER_REPLY` | Send response to customer | `message` (required) |
| `SUBMIT_RESOLUTION` | Close the ticket | `status`, `summary` |

### Action Schema (Pydantic)

```python
class Action(BaseModel):
    tool_name: Literal[
        "READ_TICKET", "GET_ORDER_DETAILS", "GET_CUSTOMER_PROFILE",
        "CHECK_POLICY", "REQUEST_APPROVAL", "EXECUTE_REFUND",
        "ISSUE_STORE_CREDIT", "SEND_CUSTOMER_REPLY", "SUBMIT_RESOLUTION"
    ]
    arguments: Dict[str, Any] = {}
    reasoning: Optional[str] = None
```

---

## 👁️ Observation Space

Each observation provides the agent with context to make decisions:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Unique task identifier |
| `ticket_text` | string | The customer support ticket content |
| `available_tools` | list[string] | Tools the agent can use |
| `last_tool_output` | dict | Output from the last tool execution |
| `workflow_history` | list[dict] | History of actions taken |
| `compliance_alerts` | list[string] | Any compliance warnings |
| `budget_remaining` | float | Remaining budget for operations |
| `max_steps_remaining` | int | Steps remaining before episode ends |
| `current_status` | string | Current ticket status |

---

## 🎮 Tasks

### Task 1: Delivery Status Resolution (Easy)

**Scenario**: Customer asks "Where is my order?"

**Expected Workflow**:
1. `READ_TICKET`
2. `GET_ORDER_DETAILS`
3. `SEND_CUSTOMER_REPLY`
4. `SUBMIT_RESOLUTION`

**Grading**:
- 1.0: Correct order status retrieved and communicated
- 0.5: Status retrieved but reply incomplete
- 0.0: Agent hallucinates or fails

---

### Task 2: Low-Value Refund with Policy Check (Medium)

**Scenario**: Customer requests refund for damaged item under $100 threshold.

**Expected Workflow**:
1. `READ_TICKET`
2. `GET_ORDER_DETAILS`
3. `CHECK_POLICY`
4. `EXECUTE_REFUND`
5. `SEND_CUSTOMER_REPLY`
6. `SUBMIT_RESOLUTION`

**Grading**:
- 1.0: Refund executed after proper policy check
- 0.7: Refund correct but reply weak
- 0.5: Refund executed without policy check
- 0.0: Denied incorrectly or failed

---

### Task 3: High-Value Refund Requiring Approval (Hard)

**Scenario**: VIP customer requests refund for expensive defective product ($599.99).

**Expected Workflow**:
1. `READ_TICKET`
2. `GET_ORDER_DETAILS`
3. `GET_CUSTOMER_PROFILE`
4. `CHECK_POLICY`
5. `REQUEST_APPROVAL`
6. `EXECUTE_REFUND`
7. `SEND_CUSTOMER_REPLY`
8. `SUBMIT_RESOLUTION`

**Grading**:
- 1.0: Full approval flow correct
- 0.6: Workflow mostly correct but messaging weak
- 0.2: Refund happens after unsafe sequence
- 0.0: Approval bypassed or task fails

---

## 💰 Reward Design

The environment provides **dense rewards** throughout the episode:

### Positive Rewards
| Action | Reward |
|--------|--------|
| First correct ticket read | +0.05 |
| Correct order retrieval | +0.10 |
| Correct customer profile retrieval | +0.08 |
| Policy check completed | +0.10 |
| Approval requested when required | +0.15 |
| Correct refund execution | +0.20 |
| Store credit issued | +0.15 |
| Customer reply sent | +0.10 |
| Resolution submitted | +0.30 |

### Penalties
| Violation | Penalty |
|-----------|---------|
| Duplicate tool call | -0.05 |
| Irrelevant tool | -0.10 |
| Wrong action ordering | -0.15 |
| Policy violation | -0.30 |
| Approval bypass | -0.40 |
| Invalid action | -0.20 |
| Extra step after optimal | -0.02 |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized deployment)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/openenv-opsflow.git
cd openenv-opsflow

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Server

```bash
# Start the FastAPI server
python app.py

# Server will be available at http://localhost:7860
```

### Running Inference

```bash
# Set required environment variables
export OPENAI_API_KEY=your-key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini

# Run baseline inference
python inference.py
```

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t openenv-opsflow .

# Run the container
docker run -p 7860:7860 openenv-opsflow

# Run with environment variables for inference
docker run -p 7860:7860 \
    -e OPENAI_API_KEY=your-key \
    -e API_BASE_URL=https://api.openai.com/v1 \
    -e MODEL_NAME=gpt-4o-mini \
    openenv-opsflow
```

---

## 🤗 Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face with Docker SDK
2. Upload all project files
3. Set the following secrets in Space settings:
   - `OPENAI_API_KEY`
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
4. The Space will automatically build and deploy

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment (optional: `task_id`) |
| `/step` | POST | Execute action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List available tasks |
| `/tasks/{id}` | GET | Get task info |

### Example Usage

```python
import requests

# Reset environment
response = requests.post("http://localhost:7860/reset", json={"task_id": "task_easy_delivery"})
observation = response.json()["observation"]

# Execute action
action = {
    "action": {
        "tool_name": "READ_TICKET",
        "arguments": {},
        "reasoning": "Starting by reading the ticket"
    }
}
response = requests.post("http://localhost:7860/step", json=action)
result = response.json()
```

---

## 📊 Baseline Scores

Tested with `gpt-4o-mini`:

| Task | Difficulty | Score | Steps |
|------|------------|-------|-------|
| task_easy_delivery | Easy | 0.95 | 4 |
| task_medium_refund | Medium | 0.85 | 6 |
| task_hard_approval | Hard | 0.75 | 8 |
| **Average** | - | **0.85** | - |

*Scores may vary slightly based on API response variability.*

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_env.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

---

## 📁 Project Structure

```
openenv-opsflow/
├── app.py              # FastAPI server
├── env.py              # Main environment class
├── models.py           # Pydantic models
├── rewards.py          # Reward calculation
├── graders.py          # Task graders
├── inference.py        # Baseline inference script
├── openenv.yaml        # OpenEnv specification
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── .env.example        # Environment variables template
├── data/
│   ├── orders.json     # Mock order data
│   ├── customers.json  # Mock customer data
│   └── policies.json   # Mock policy data
├── tasks/
│   └── __init__.py     # Task definitions
├── tests/
│   ├── test_env.py     # Environment tests
│   ├── test_graders.py # Grader tests
│   └── test_api.py     # API tests
└── utils/
    └── __init__.py
```

---

## 🔒 Compliance & Safety Features

- **Approval-gated workflows**: High-value actions require explicit approval
- **Fraud detection**: Blocks refunds to fraud-flagged customers
- **Policy enforcement**: Actions validated against business rules
- **Audit logging**: Complete action history maintained
- **Step limits**: Prevents infinite loops

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built for the Meta OpenEnv Hackathon. This environment demonstrates how AI agents can be trained and evaluated on real-world enterprise workflows.
