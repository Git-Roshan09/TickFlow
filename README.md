<div align="center">

<img src="assets/logo.png" alt="TickFlow Logo" width="400"/>

### Policy-Aware Enterprise Support Orchestration Environment

**The first OpenEnv environment with approval gates, audit trail scoring, and compliance enforcement**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-00D4AA?style=for-the-badge&logo=openai&logoColor=white)](https://github.com/openenv)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Hugging Face](https://img.shields.io/badge/HF-Spaces-FFD21E?style=for-the-badge)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

[**Live Demo**](https://huggingface.co/spaces/YOUR_USERNAME/tickflow-openenv) · [**Documentation**](#quick-start) · [**API Reference**](#api-usage) · [**Benchmarks**](#baseline-scores)

</div>

---

## 🏅 Novel Contributions — What Makes TickFlow Different

<table>
<tr>
<td align="center" width="25%">
<h3>🔐</h3>
<b>Approval Gates</b><br/>
<sub>First OpenEnv with human-in-the-loop simulation. High-value actions require supervisor approval.</sub>
</td>
<td align="center" width="25%">
<h3>📋</h3>
<b>Audit Trail Scoring</b><br/>
<sub>Agents rewarded for explainable, reviewable workflows — not just correct answers.</sub>
</td>
<td align="center" width="25%">
<h3>🛡️</h3>
<b>Compliance Enforcement</b><br/>
<sub>Policy violations automatically detected and penalized with deterministic grading.</sub>
</td>
<td align="center" width="25%">
<h3>📊</h3>
<b>Multi-Component Rewards</b><br/>
<sub>Dense rewards: outcome + efficiency + compliance + audit quality - privacy overreach.</sub>
</td>
</tr>
</table>


> **Why this matters**: Enterprise AI platforms require audit trails, explainability, approval checkpoints, and compliance-aware orchestration. TickFlow is the first environment that evaluates ALL of these in a unified benchmark.

---

## Real-World Enterprise Problem

**Customer support is a $50B+ industry** where AI agents must orchestrate 8+ internal systems:

| System | Purpose | Risk if Misused | TickFlow Tool |
|--------|---------|-----------------|---------------|
| CRM | Customer records | Privacy violation | `GET_CUSTOMER_PROFILE` |
| Order Database | Purchase history | Data leakage | `GET_ORDER_DETAILS` |
| Fraud Detection | Risk assessment | False accusations | Built-in checks |
| Policy Engine | Business rules | Compliance breach | `CHECK_POLICY` |
| Approval System | Authorization | Unauthorized actions | `REQUEST_APPROVAL` |
| Payment Gateway | Refunds/credits | Financial loss | `EXECUTE_REFUND` |
| Notification | Customer comms | Reputation damage | `SEND_CUSTOMER_REPLY` |
| Audit Log | Compliance trail | Legal liability | Full audit trail |

**The challenge**: An AI agent must orchestrate these systems in the correct order, following business policies, obtaining approvals when required, and maintaining audit trails — while resolving customer issues efficiently.

---

## Key Innovation Details

<table>
<tr>
<td width="50%">

### Approval-Gated Actions
High-value refunds (>$100) require supervisor approval before execution. The agent must recognize when to escalate.

```python
# Agent tries to refund $599 without approval
action = Action(tool_name="EXECUTE_REFUND")
# Reward: -0.40 (compliance violation)

# Agent requests approval first  
action = Action(tool_name="REQUEST_APPROVAL")
# Reward: +0.15 (correct escalation)
```

</td>
<td width="50%">

### Audit Trail Scoring
Every action is logged. Agents are rewarded for clean, justifiable workflows that a human auditor could review.

```python
audit_log = [
    {"step": 1, "tool": "READ_TICKET", "reason": "..."},
    {"step": 2, "tool": "GET_ORDER", "reason": "..."},
    {"step": 3, "tool": "CHECK_POLICY", "reason": "..."},
]
# Audit quality contributes to final score
```

</td>
</tr>
<tr>
<td width="50%">

### Compliance Enforcement
Policy violations are automatically detected and penalized. Skip a required step? Instant penalty.

```
Violation: refund_without_approval  → Penalty: -0.40
Violation: order_lookup_before_ticket → Penalty: -0.15
Violation: skip_policy_check → Penalty: -0.30
```

</td>
<td width="50%">

### Dense Reward Shaping
Not just pass/fail — every step matters. Partial progress is rewarded, inefficiency is penalized.

```
R_step = R_tool + R_order + R_compliance + R_efficiency
```

</td>
</tr>
</table>

---

## Architecture

<div align="center">
<img src="assets/flow.png" alt="flow" width="100%"/>
</div>

---


<div align="center">
<img src="assets/opsflow_architecture.png" alt="TickFlow System Architecture" width="100%"/>
</div>

*Figure 1: System Architecture — Policy-Aware Enterprise Support Orchestration Environment showing Agent Layer, Tool Registry (9 tools), State Manager, and Deterministic Grader.*

---

## OpenEnv Core Interface (`reset`, `step`, `state`)

TickFlow follows the standard OpenEnv interaction loop:

```python
# 1) Start new episode
observation = reset(task_id="task_easy_delivery")

# 2) Agent acts until done=True
while True:
    action = {
        "tool_name": "READ_TICKET",
        "arguments": {},
        "reasoning": "Read the ticket first"
    }
    observation, reward, done, info = step(action)
    if done:
        break

# 3) Inspect full internal state (debugging/evaluation)
current_state = state()
```

`reset()` returns the initial observation for the selected task, `step()` applies one action and returns `(observation, reward, done, info)`, and `state()` returns the complete environment state (workflow flags, violations, audit trail, and reward totals).

### Endpoint Mapping

| OpenEnv Method | HTTP Endpoint | Method | Purpose |
|----------------|---------------|--------|---------|
| `reset(task_id)` | `/reset` | POST | Start episode and return initial observation |
| `step(action)` | `/step` | POST | Execute one action and return transition |
| `state()` | `/state` | GET | Return current full environment state |

---

## Evaluation Criteria Alignment

| Criteria | Weight | Evidence |
|----------|--------|----------|
| **Real-world utility** | 30% | $50B+ enterprise support industry; models actual business workflows |
| **Task & grader quality** | 25% | 3 tasks (easy→hard), deterministic graders, scores in [0.0, 1.0] |
| **Environment design** | 20% | Clean state management, approval gates, dense reward shaping |
| **Code quality** | 15% | Typed Pydantic models, tested, documented, Dockerfile works |
| **Creativity & novelty** | 10% | First approval-gated, audit-aware, compliance-enforcing environment |

---

## Tasks

<div align="center">
<img src="assets/opsflow_workflow.png" alt="Agent Workflow Paths" width="100%"/>
</div>

*Figure 2: Agent Workflow Paths — Three escalating tasks, each testing deeper compliance and approval logic.*

### Task Summary

| Task | Difficulty | Steps | Key Challenge | Baseline Score |
|------|------------|-------|---------------|----------------|
| **Delivery Status Inquiry** | Easy | 4 | Simple retrieval + response | 0.95 |
| **Low-Value Refund** | Medium | 6 | Must `CHECK_POLICY` before refund | 0.85 |
| **VIP High-Value Refund** | Hard | 8 | Requires `REQUEST_APPROVAL` (>$100) | 0.75 |

### Why Hard Task Challenges Frontier Models

The hard task requires the agent to:
1. Recognize VIP status from customer profile
2. Identify high-value threshold ($599.99 > $100)
3. Request approval BEFORE refund (not after)
4. Maintain audit trail with reasoning
5. Handle approval response correctly

Bypassing approval results in **score = 0.0** (compliance failure)

---

## Reward System

<div align="center">
<img src="assets/opsflow_reward_compliance.png" alt="Reward Design & Compliance System" width="100%"/>
</div>

*Figure 3: Reward Design & Compliance System — Dense multi-component rewards with deterministic violation detection.*

### Mathematical Formulation

**Step Reward Function:**

$$
R_t =
\underbrace{R_{\text{tool}}(a_t)}_{\text{correct tool usage}}
\;+\;
\underbrace{R_{\text{order}}(a_t, s_t)}_{\text{workflow order correctness}}
\;+\;
\underbrace{R_{\text{comp}}(a_t, s_t)}_{\text{policy compliance}}
\;+\;
\underbrace{R_{\text{eff}}(t)}_{\text{efficiency term}}
$$

Where:
- $R_{\text{tool}}(a_t)$ = reward for executing tool $a_t$ correctly  
- $R_{\text{order}}(a_t, s_t)$ = penalty if action violates required ordering given state $s_t$  
- $R_{\text{comp}}(a_t, s_t)$ = policy violation penalty (e.g., bypassing approval)  
- $R_{\text{eff}}(t)= -0.02 \cdot \max(0, t - T^\*)$, where $T^\*$ is the optimal step count

**Episode Score (Grading):**

$$
S_{\text{episode}} =
\operatorname{clip}_{[0,1]}
\left(
\frac{\sum_{t=1}^{T} R_t + 0.5}{1.5}
\right)
$$

**Grader Score Function (Task-specific):**

$$
G(s_{\text{final}}) =
\operatorname{clip}_{[0,1]}
\left(
\sum_i w_i \,\mathbf{1}[\text{condition}_i]
-\sum_j p_j \, |\text{violations}_j|
\right)
$$

Where `w_i` are component weights and `p_j` are violation penalties.

### Reward Constants

| Action | Reward | Description |
|--------|--------|-------------|
| `READ_TICKET` | +0.05 | Initial ticket comprehension |
| `GET_ORDER_DETAILS` | +0.10 | Order data retrieval (after ticket read) |
| `GET_CUSTOMER_PROFILE` | +0.08 | Customer data retrieval |
| `CHECK_POLICY` | +0.10 | Policy verification |
| `REQUEST_APPROVAL` | +0.15 | Escalation to supervisor |
| `EXECUTE_REFUND` | +0.20 | Refund execution (after approval if required) |
| `SEND_CUSTOMER_REPLY` | +0.10 | Customer communication |
| `SUBMIT_RESOLUTION` | +0.30 | Ticket closure |

### Penalty Constants

| Violation | Penalty | Severity |
|-----------|---------|----------|
| Duplicate tool call | -0.05 | Low |
| Irrelevant tool | -0.10 | Low |
| Wrong action order | -0.15 | Medium |
| Skip policy check | -0.30 | High |
| **Bypass approval** | **-0.40** | Critical |
| Invalid action | -0.20 | Medium |
| Extra step beyond optimal | -0.02 | Minor |

---

## Technical Specification

### Action Space

```python
class Action(BaseModel):
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
    ]
    arguments: Dict[str, Any] = {}
    reasoning: Optional[str] = None
```

### Observation Space

```python
class Observation(BaseModel):
    task_id: str                    # Current task identifier
    ticket_text: str                # Customer's message
    available_tools: List[str]      # Tools agent can use
    last_tool_output: Dict          # Result of last action
    workflow_history: List[Dict]    # Actions taken so far
    compliance_alerts: List[str]    # Any violations
    max_steps_remaining: int        # Steps before timeout
    current_status: str             # Workflow state
```

### Environment State

```python
class EnvironmentState(BaseModel):
    # Task info
    task_id: str
    task_difficulty: str
    
    # Workflow tracking
    ticket_read: bool
    order_retrieved: Optional[Dict]
    customer_retrieved: Optional[Dict]
    policy_checked: bool
    approval_requested: bool
    approval_status: Optional[str]
    refund_executed: bool
    customer_reply_sent: bool
    resolution_submitted: bool
    
    # Audit trail
    action_history: List[Dict]
    compliance_violations: List[str]
    total_reward: float
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/tickflow.git
cd tickflow

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### Run Locally

```bash
python app.py
# Server runs at http://localhost:7860
```

### Docker

```bash
docker build -t tickflow .
docker run -p 7860:7860 tickflow
```

---

## API Usage

### Reset Environment

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy_delivery"}'
```

### Take Action

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "tool_name": "READ_TICKET",
      "arguments": {},
      "reasoning": "Starting by understanding the customer request"
    }
  }'
```

### Get State

```bash
curl http://localhost:7860/state
```

### List Tasks

```bash
curl http://localhost:7860/tasks
```

---

## Baseline Scores

Tested with `Qwen/Qwen2.5-72B-Instruct`:

| Task | Difficulty | Score | Steps | Status |
|------|------------|-------|-------|--------|
| `task_easy_delivery` | Easy | **0.95** | 4 | Pass |
| `task_medium_refund` | Medium | **0.85** | 6 | Pass |
| `task_hard_approval` | Hard | **0.75** | 8 | Pass |
| **Average** | - | **0.85** | - | - |

---


## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

---

## Hugging Face Deployment

1. Create a new Space with **Docker SDK**
2. Upload all project files
3. Set secrets:
   - `HF_TOKEN`
   - `API_BASE_URL`
   - `MODEL_NAME`
4. Wait for build (~3 minutes)

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built for the **Meta OpenEnv Hackathon**. 

Special thanks to the OpenEnv team for creating a framework that enables production-oriented AI evaluation.

---

<div align="center">

**[Back to Top](#policy-aware-enterprise-support-orchestration-environment)**

Made by Roshan & Baranidharan

</div>
