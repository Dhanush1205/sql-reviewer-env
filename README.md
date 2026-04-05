# SQL Reviewer — OpenEnv Environment

An RL environment where an AI agent acts as a **SQL code reviewer**, catching syntax errors, security vulnerabilities, and performance anti-patterns in queries written by junior developers.

---

## Why This Environment?

SQL quality is a real, high-stakes problem at every tech company. Bad SQL causes outages, data breaches, and slow applications. This environment trains agents to catch issues that humans commonly miss under time pressure.

---

## Tasks

| Task ID | Difficulty | Goal |
|---|---|---|
| `syntax_fix` | Easy | Fix 3 keyword typos in a SELECT query |
| `sql_injection` | Medium | Detect & parameterize a SQL injection vulnerability |
| `query_optimization` | Hard | Eliminate N+1, add aggregation, fix date filters |

---

## Action Space

```json
{
  "review_comment": "string — explanation of all issues found",
  "fixed_query":    "string — corrected SQL query",
  "issues_found":   ["string", ...],
  "severity":       "low | medium | high | critical"
}
```

## Observation Space

```json
{
  "task_id":           "string",
  "task_description":  "string",
  "sql_query":         "string — the buggy query to review",
  "schema_context":    "string — table definitions",
  "step_number":       "int",
  "previous_feedback": "string | null",
  "hints":             ["string", ...]
}
```

## Reward

Dense (not sparse) — partial credit awarded at every step across:
- **Keyword correction** (syntax task)
- **Vulnerability detection + parameterization** (injection task)
- **JOIN, aggregation, date filter, index advice** (optimization task)

Score range: `0.0 – 1.0`. Episode ends when score ≥ 0.85 or max 5 steps reached.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/sql-reviewer-env
cd sql-reviewer-env
pip install -r requirements.txt
```

### Run server
```bash
python app.py
```

### Run inference
```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Docker
```bash
docker build -t sql-reviewer-env .
docker run -p 7860:7860 -e HF_TOKEN=your_token sql-reviewer-env
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Environment info |
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks |
| POST | `/reset?task_id=syntax_fix` | Start new episode |
| POST | `/step?task_id=syntax_fix` | Take action |
| GET | `/state?task_id=syntax_fix` | Current state |

---

## Baseline Scores

| Task | Model | Score |
|---|---|---|
| syntax_fix | Qwen2.5-72B-Instruct | ~0.90 |
| sql_injection | Qwen2.5-72B-Instruct | ~0.80 |
| query_optimization | Qwen2.5-72B-Instruct | ~0.65 |

---

## Team

**MetaAgents** — OpenEnv Hackathon 2025
