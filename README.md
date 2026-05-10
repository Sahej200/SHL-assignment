# SHL Assessment Advisor — Agent API

A stateless FastAPI service that acts as an intelligent SHL assessment recommender. Given a conversation history, it returns the next agent reply and (when appropriate) a structured shortlist of up to 10 SHL catalog assessments.

---

## Project Structure

```
shl_agent/
├── main.py                  # FastAPI app + agent logic
├── catalog.json             # SHL catalog (377 assessments scraped from shl.com)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container image
├── test_agent.py            # Behavioral test suite
├── sample_conversations/    # 10 example agent conversations (C1–C10)
└── README.md
```

---

## Quick Start

### 1. Set your API key
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Test it
```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hiring a Java developer who works with stakeholders"},
      {"role": "assistant", "content": "What seniority level are you hiring for?"},
      {"role": "user", "content": "Mid-level, about 4 years experience"}
    ]
  }'
```

---

## Docker

```bash
docker build -t shl-agent .
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY -p 8000:8000 shl-agent
```

---

## API Reference

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments that fit a mid-level Java dev...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/products/product-catalog/view/java-8-new/", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is an empty array `[]` while the agent is still clarifying.
- `end_of_conversation` is `true` only when the agent considers the task complete.
- Every URL is guaranteed to come from the scraped SHL catalog.

---

## Agent Behaviors

| Behavior | Description |
|---|---|
| **Clarify** | Asks a focused question for vague queries ("I need an assessment") |
| **Recommend** | Returns 1–10 catalog assessments once role, level, and purpose are clear |
| **Refine** | Updates the shortlist when the user changes constraints mid-conversation |
| **Compare** | Compares assessments using only catalog descriptions (no hallucination) |
| **Refuse** | Rejects off-topic requests, legal questions, and prompt-injection attempts |

---

## Catalog

The catalog (`catalog.json`) contains **377 SHL assessments** scraped from [shl.com/products/product-catalog](https://www.shl.com/products/product-catalog/), covering:

| Type | Count | Description |
|---|---|---|
| K — Knowledge & Skills | 240 | Technical skill tests (Java, Python, SQL, AWS, …) |
| P — Personality & Behavior | 66 | OPQ32r, MQ, work styles |
| S — Simulations | 43 | Coding simulations, MS Office sims |
| A — Ability & Aptitude | 32 | Cognitive/numerical/verbal reasoning |
| C — Competencies | 19 | Competency frameworks |
| B — Biodata & SJT | 17 | Situational judgment, job-focused |
| D — Development & 360 | 7 | 360 feedback tools |
| E — Assessment Exercises | 2 | Assessment center exercises |

Each catalog entry includes: `name`, `url`, `test_types`, `description`, `job_levels`, `duration`, `languages`, `remote`, and `adaptive` fields.

---

## Design Decisions

- **Stateless**: full conversation history is sent with every POST; the server stores nothing.
- **Catalog-grounded**: the system prompt embeds the full catalog; URL validation rejects any hallucinated links.
- **Turn-efficient**: the prompt instructs the model to clarify with one targeted question, not a list, to stay within the 8-turn cap.
- **Schema-safe**: JSON is parsed from a fenced block in the LLM output; the rest is plain text for `reply`.

---

## Running Tests

```bash
# Start server first in another terminal
uvicorn main:app --port 8000

# Run tests
python test_agent.py
```

Tests cover: health check, vague query clarification, Java dev recommendation, mid-conversation refinement, product comparison, off-topic refusal, and prompt injection resistance.
"# SHL-assignment" 
