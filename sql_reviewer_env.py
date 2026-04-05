"""
SQL Query Reviewer — OpenEnv Environment
=========================================
An RL environment where an AI agent reviews SQL queries
submitted by junior developers, checking for:
  - Syntax errors  (easy)
  - SQL injection vulnerabilities  (medium)
  - Performance / optimization issues  (hard)
"""

import re
from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────
# Typed Models  (OpenEnv spec)
# ─────────────────────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    task_description: str
    sql_query: str
    schema_context: str
    step_number: int
    previous_feedback: Optional[str] = None
    hints: list[str] = Field(default_factory=list)


class Action(BaseModel):
    review_comment: str        # Agent's explanation of what's wrong
    fixed_query: str           # Agent's corrected SQL
    issues_found: list[str]    # List of identified issues
    severity: str              # "low" | "medium" | "high" | "critical"


class Reward(BaseModel):
    score: float               # 0.0 – 1.0
    breakdown: dict[str, float]
    feedback: str


# ─────────────────────────────────────────────────────────────
# Task Definitions
# ─────────────────────────────────────────────────────────────

TASKS = {
    "syntax_fix": {
        "id": "syntax_fix",
        "difficulty": "easy",
        "description": (
            "A junior developer wrote a SQL query to fetch all active users "
            "from the 'users' table who registered after 2022. "
            "The query has syntax errors. Find and fix ALL of them."
        ),
        "schema": (
            "Table: users\n"
            "  - id INTEGER PRIMARY KEY\n"
            "  - name TEXT NOT NULL\n"
            "  - email TEXT UNIQUE\n"
            "  - status TEXT  -- 'active' or 'inactive'\n"
            "  - created_at DATETIME\n"
        ),
        "buggy_query": (
            "SELCT name, email FORM users\n"
            "WHER status = 'active' AND created_at > '2022-01-01'\n"
            "ORDER BY created_at DEC;"
        ),
        "correct_keywords": ["SELECT", "FROM", "WHERE", "DESC"],
        "expected_fix_pattern": r"SELECT.+FROM\s+users.+WHERE.+status.+ORDER BY.+DESC",
        "hints": [
            "Look carefully at every keyword spelling.",
            "Check the ORDER BY direction keyword.",
        ],
    },

    "sql_injection": {
        "id": "sql_injection",
        "difficulty": "medium",
        "description": (
            "A junior developer wrote a login query that concatenates user input "
            "directly into the SQL string. This is critically dangerous! "
            "Identify the SQL injection vulnerability and rewrite the query "
            "using parameterized inputs."
        ),
        "schema": (
            "Table: users\n"
            "  - id INTEGER PRIMARY KEY\n"
            "  - username TEXT UNIQUE\n"
            "  - password_hash TEXT\n"
            "  - role TEXT  -- 'admin' or 'user'\n"
        ),
        "buggy_query": (
            'query = "SELECT * FROM users WHERE username = \'" + username + "\' AND password_hash = \'" + password + "\'"'
        ),
        "correct_keywords": ["?", "%s", "parameterized", "placeholder", "prepared"],
        "expected_fix_pattern": r"(\?|%s|:username|:password|:1|:2)",
        "hints": [
            "Never concatenate user input into SQL strings.",
            "Use placeholders like ? or %s with parameter binding.",
        ],
    },

    "query_optimization": {
        "id": "query_optimization",
        "difficulty": "hard",
        "description": (
            "A production query is running very slowly on a 10-million-row table. "
            "It fetches order totals per customer for the current month. "
            "Identify ALL performance issues (N+1 pattern, missing index, "
            "function on indexed column, SELECT *, no aggregation) "
            "and rewrite it as a single optimized SQL query."
        ),
        "schema": (
            "Table: orders\n"
            "  - id INTEGER PRIMARY KEY\n"
            "  - customer_id INTEGER  (FK → customers.id)\n"
            "  - amount DECIMAL(10,2)\n"
            "  - status TEXT\n"
            "  - created_at DATETIME\n\n"
            "Table: customers\n"
            "  - id INTEGER PRIMARY KEY\n"
            "  - name TEXT\n"
            "  - email TEXT\n"
            "  - country TEXT\n"
        ),
        "buggy_query": (
            "SELECT * FROM orders WHERE\n"
            "  YEAR(created_at) = YEAR(NOW()) AND\n"
            "  MONTH(created_at) = MONTH(NOW()) AND\n"
            "  status != 'cancelled';\n\n"
            "-- Then in application code:\n"
            "-- for order in orders:\n"
            "--     customer = db.query(f'SELECT * FROM customers WHERE id = {order.customer_id}')\n"
            "--     total += order.amount\n"
        ),
        "correct_keywords": ["JOIN", "GROUP BY", "SUM", "created_at >="],
        "expected_fix_pattern": r"(JOIN|SUM\s*\(|GROUP BY|created_at\s*>=)",
        "hints": [
            "Avoid applying functions to indexed columns in WHERE clauses.",
            "Eliminate the N+1 query loop with a JOIN.",
            "Use SUM() with GROUP BY instead of summing in application code.",
            "Suggest CREATE INDEX on created_at and customer_id.",
        ],
    },
}


# ─────────────────────────────────────────────────────────────
# Graders  (deterministic, 0.0–1.0)
# ─────────────────────────────────────────────────────────────

def _grade_syntax_fix(action: Action, task: dict) -> Reward:
    score = 0.0
    breakdown: dict[str, float] = {}

    fixed_upper = action.fixed_query.upper()

    # 40% — all four keywords corrected
    kw_hits = sum(1 for kw in task["correct_keywords"] if kw in fixed_upper)
    breakdown["keyword_correction"] = round(kw_hits / len(task["correct_keywords"]) * 0.40, 3)

    # 30% — regex structure looks like a valid SELECT … FROM … WHERE … ORDER BY … DESC
    pattern_ok = bool(re.search(task["expected_fix_pattern"], action.fixed_query, re.IGNORECASE | re.DOTALL))
    breakdown["structural_correctness"] = 0.30 if pattern_ok else 0.0

    # 20% — agent named at least 2 concrete issues
    issues_blob = " ".join(action.issues_found).lower()
    spotted = sum(w in issues_blob for w in ["selct", "form", "wher", "dec", "syntax", "typo", "spelling", "keyword"])
    breakdown["issues_identified"] = round(min(spotted / 2, 1.0) * 0.20, 3)

    # 10% — non-trivial review comment
    breakdown["review_quality"] = 0.10 if len(action.review_comment) > 40 else 0.0

    score = round(sum(breakdown.values()), 3)
    feedback = (
        "Excellent — all syntax errors found and fixed!" if score >= 0.8
        else "Partial fix — some issues remain." if score >= 0.5
        else "Incomplete. Re-read the query keyword by keyword."
    )
    return Reward(score=score, breakdown=breakdown, feedback=feedback)


def _grade_sql_injection(action: Action, task: dict) -> Reward:
    score = 0.0
    breakdown: dict[str, float] = {}

    fixed = action.fixed_query
    blob = (action.review_comment + " " + " ".join(action.issues_found)).lower()

    # 30% — vulnerability explicitly named
    breakdown["vulnerability_detected"] = 0.30 if any(
        w in blob for w in ["injection", "concatenat", "unsafe", "vulnerable"]
    ) else 0.0

    # 40% — fix uses a parameterised placeholder
    breakdown["parameterized_query"] = 0.40 if re.search(task["expected_fix_pattern"], fixed) else 0.0

    # 20% — no string concatenation survives in the fix
    concat_gone = ("' +" not in fixed) and ('+ "' not in fixed) and ("+ '" not in fixed)
    breakdown["no_concatenation"] = 0.20 if concat_gone else 0.0

    # 10% — severity correctly flagged
    breakdown["severity_rating"] = 0.10 if action.severity in ("high", "critical") else 0.0

    score = round(sum(breakdown.values()), 3)
    feedback = (
        "Perfect — injection caught and parameterised!" if score >= 0.8
        else "Partially addressed — ensure no concatenation remains and placeholders are used." if score >= 0.5
        else "SQL injection vulnerability not properly addressed."
    )
    return Reward(score=score, breakdown=breakdown, feedback=feedback)


def _grade_query_optimization(action: Action, task: dict) -> Reward:
    score = 0.0
    breakdown: dict[str, float] = {}

    fixed_upper = action.fixed_query.upper()
    blob = (action.review_comment + " " + " ".join(action.issues_found)).lower()

    # 25% — N+1 eliminated via JOIN
    breakdown["eliminated_n_plus_1"] = 0.25 if "JOIN" in fixed_upper else 0.0

    # 25% — proper aggregation
    agg_ok = "SUM(" in fixed_upper and "GROUP BY" in fixed_upper
    breakdown["proper_aggregation"] = 0.25 if agg_ok else 0.0

    # 20% — range filter instead of function-on-column
    range_ok = ("CREATED_AT >=" in fixed_upper) or ("CREATED_AT BETWEEN" in fixed_upper)
    breakdown["fixed_date_filter"] = 0.20 if range_ok else 0.0

    # 15% — indexing mentioned
    breakdown["index_recommendation"] = 0.15 if any(w in blob for w in ["index", "indices", "indexed"]) else 0.0

    # 15% — SELECT * removed
    breakdown["no_select_star"] = 0.15 if "SELECT *" not in fixed_upper else 0.0

    score = round(min(sum(breakdown.values()), 1.0), 3)
    feedback = (
        "Outstanding — all performance issues resolved!" if score >= 0.8
        else "Good progress but some optimizations are still missing." if score >= 0.5
        else "The query still has significant performance problems."
    )
    return Reward(score=score, breakdown=breakdown, feedback=feedback)


GRADERS = {
    "syntax_fix": _grade_syntax_fix,
    "sql_injection": _grade_sql_injection,
    "query_optimization": _grade_query_optimization,
}


# ─────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────

class SQLReviewerEnv:
    """OpenEnv-compliant SQL Query Reviewer Environment."""

    MAX_STEPS = 5

    def __init__(self, task_id: str = "syntax_fix"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task '{task_id}'. Valid: {list(TASKS)}")
        self.task_id = task_id
        self._task = TASKS[task_id]
        self._step = 0
        self._done = False
        self._last_reward: Optional[Reward] = None
        self._history: list[dict] = []

    # ── OpenEnv API ───────────────────────────────────────────

    def reset(self) -> Observation:
        self._step = 0
        self._done = False
        self._last_reward = None
        self._history = []
        return self._obs()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._done:
            raise RuntimeError("Episode done. Call reset().")
        self._step += 1
        reward = GRADERS[self.task_id](action, self._task)
        self._last_reward = reward
        self._history.append({"step": self._step, "action": action.model_dump(), "reward": reward.model_dump()})
        self._done = self._step >= self.MAX_STEPS or reward.score >= 0.85
        done_reason = "success" if reward.score >= 0.85 else ("max_steps" if self._done else "ongoing")
        info = {"step": self._step, "score": reward.score, "breakdown": reward.breakdown, "done_reason": done_reason}
        return self._obs(reward.feedback), reward, self._done, info

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step": self._step,
            "done": self._done,
            "last_score": self._last_reward.score if self._last_reward else None,
            "history": self._history,
        }

    def close(self):
        pass

    # ── Internal ──────────────────────────────────────────────

    def _obs(self, previous_feedback: Optional[str] = None) -> Observation:
        return Observation(
            task_id=self.task_id,
            task_description=self._task["description"],
            sql_query=self._task["buggy_query"],
            schema_context=self._task["schema"],
            step_number=self._step,
            previous_feedback=previous_feedback,
            hints=self._task["hints"] if self._step == 0 else [],
        )
