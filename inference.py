"""
Inference Script — SQL Reviewer OpenEnv
========================================
Runs an LLM agent against all 3 tasks and emits structured stdout logs.

MANDATORY env vars:
  API_BASE_URL   — LLM API endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME     — model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       — HuggingFace / API key  (no default — must be set)

STDOUT FORMAT (exact):
  [START] task=<task_name> env=sql-reviewer model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import sys

from openai import OpenAI

from sql_reviewer_env import SQLReviewerEnv, Action, TASKS

# ── Configuration (checklist item 2 & 3) ──────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")          # no default — must be provided

BENCHMARK  = "sql-reviewer"
MAX_STEPS  = 5

# ── OpenAI client (checklist item 4) ──────────────────────────────────────────
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert SQL code reviewer. You will be given a SQL query with problems.
Your job is to:
1. Identify all issues in the query
2. Explain what is wrong (review_comment)
3. Provide a corrected version (fixed_query)
4. List specific issues found (issues_found)
5. Rate severity: low | medium | high | critical

You MUST respond with ONLY valid JSON matching this exact schema:
{
  "review_comment": "<detailed explanation>",
  "fixed_query": "<corrected SQL>",
  "issues_found": ["<issue1>", "<issue2>", ...],
  "severity": "<low|medium|high|critical>"
}
No extra text, no markdown, no code fences — pure JSON only."""


def build_user_prompt(obs) -> str:
    prompt = f"""Task: {obs.task_description}

Schema:
{obs.schema_context}

Buggy SQL Query:
{obs.sql_query}
"""
    if obs.previous_feedback:
        prompt += f"\nPrevious feedback: {obs.previous_feedback}\n"
    if obs.hints:
        prompt += f"\nHints: {', '.join(obs.hints)}\n"
    prompt += "\nReview the query and respond with JSON only."
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Agent step
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(obs) -> Action:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(obs)},
        ],
        temperature=0.2,
        max_tokens=1000,
    )
    raw = response.choices[0].message.content.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    return Action(**data)


# ─────────────────────────────────────────────────────────────────────────────
# Run one task episode
# ─────────────────────────────────────────────────────────────────────────────

def run_task(task_id: str) -> float:
    env   = SQLReviewerEnv(task_id=task_id)
    obs   = env.reset()
    done  = False
    step  = 0
    rewards: list[float] = []
    last_error = None
    final_score = 0.0

    # ── [START] ──────────────────────────────────────────────────────────────
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        while not done and step < MAX_STEPS:
            step += 1
            last_error = None

            try:
                action = call_llm(obs)
                action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            except Exception as e:
                last_error = str(e).replace("\n", " ")
                action_str = "null"
                # emit STEP with error, then end
                print(
                    f"[STEP] step={step} action={action_str} reward=0.00 done=true error={last_error}",
                    flush=True,
                )
                done = True
                break

            obs, reward, done, info = env.step(action)
            rewards.append(reward.score)
            final_score = reward.score

            error_field = last_error if last_error else "null"
            done_str    = "true" if done else "false"

            # ── [STEP] ───────────────────────────────────────────────────────
            print(
                f"[STEP] step={step} action={action_str} "
                f"reward={reward.score:.2f} done={done_str} error={error_field}",
                flush=True,
            )

        env.close()

    except Exception as e:
        last_error = str(e).replace("\n", " ")
        done = True

    success     = final_score >= 0.85
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    success_str = "true" if success else "false"

    # ── [END] ────────────────────────────────────────────────────────────────
    print(
        f"[END] success={success_str} steps={step} score={final_score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return final_score


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    all_scores: list[float] = []
    for task_id in TASKS:
        score = run_task(task_id)
        all_scores.append(score)

    avg = sum(all_scores) / len(all_scores)
    print(f"\nOverall average score: {avg:.2f}", flush=True)
