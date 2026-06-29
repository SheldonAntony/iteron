MENTAL_SIMULATOR_TEMPLATE = """\
System: [What are we optimizing?]
Structural Analogue:
  Core components: [list]
  Functional interactions: [how X affects Y]
  Causal constraints: [if I change A, B must happen because of principle C]
Hypothesis Space:
  Current best hypothesis: [what we think works]
  Competing hypothesis: [alternative explanation]
  Key differentiator: [what experiment would distinguish them]
Before writing code, I must:
  1. Define the structural analogue
  2. Identify causal constraints
  3. Propose a hypothesis with a falsifiable prediction
  4. Only then: Write Python implementation."""

PROBLEMATIZER_PROMPT = """\
You are in PROBLEMATIZER MODE.
Your job is NOT to improve the metric. Your job is to challenge assumptions.

Identify:
  Field Assumptions: What does the field take for granted?
  In-House Assumptions: What does our codebase assume?
  What if the opposite were true?

Propose 3 experiments that would falsify each assumption.

For each experiment, specify:
  - What assumption it tests
  - What result would falsify that assumption
  - Expected cost in compute/time

Current problem: {problem}
Current best solution code:
```python
{current_code}
```"""

META_AGENT_PROMPT = """\
You are the ITERON Meta Agent. Your job is to read the self-model and propose
ONE structural change to improve performance.

Current self-model:
{self_model}

Rules:
- Propose exactly ONE change.
- The change must be specific and falsifiable.
- Specify: what to change, from what to what, and why.
- Estimate the expected impact on CPG.
- After your proposal, write a 3-line human-readable summary starting with 'SUMMARY:'.

Examples of valid changes:
- "Increase plateau_window from 5 to 8 to reduce false positives"
- "Switch experiment tags from auto to manual H-space after round 10"
- "Reduce anomaly threshold from 2.0 to 1.5 sigma for earlier detection"
- "Route problematizer calls from smart tier to fast tier to save budget"
"""

ANOMALY_DEBATE_A = """\
You are Agent A in an anomaly debate.

Score {score:.4f} was observed (expected ~{mean:.4f}).

Your position: This anomaly is GENUINE and means something important.
Argue why this result is real, not noise. What mechanism explains it?
Consider: could this be a real improvement, a hidden bug, or a new phenomenon?

Problem context: {problem}"""

ANOMALY_DEBATE_B = """\
You are Agent B in an anomaly debate.

Score {score:.4f} was observed (expected ~{mean:.4f}).

Your position: This anomaly is NOISE or measurement error.
Argue why this result should be dismissed. Consider: fluke, non-determinism,
overfitting, broken eval, or environmental factors.

Problem context: {problem}"""


def build_proposal_prompt(
    problem: str, current_code: str, dmf_context: str
) -> tuple[str, str]:
    system = (
        f"You are ITERON, a research agent optimizing: {problem}\n\n"
        "Rules:\n"
        "1. Output ONLY valid Python code inside ```python``` blocks.\n"
        "2. The code MUST be syntactically valid and run without errors.\n"
        "3. Make small targeted changes to the module-level constants "
        "(VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, EXPANSION_FACTOR, "
        "DROPOUT, ACTIVATION, LEARNING_RATE, BATCH_SIZE, TRAIN_STEPS, "
        "SEQ_LEN, WARMUP_STEPS) unless there's a proven reason to change "
        "the architecture.\n"
        "4. Keep total params under 1,000,000.\n"
        "5. Output the COMPLETE solution.py file, including all imports.\n"
        "6. ABSOLUTELY no markdown outside the code block."
    )
    prompt = (
        f"Current best solution:\n"
        f"```python\n{current_code}\n```\n\n"
        f"{dmf_context}\n\n"
        "Propose an improved version. Output ONLY the code in a ```python``` block."
    )
    return system, prompt


def build_cold_start_prompt(problem: str) -> str:
    return (
        f"You are ITERON. Write a Python solution for this problem:\n\n"
        f"{problem}\n\n"
        "Write complete, runnable code. Include all necessary imports and functions.\n"
        "Output ONLY the code inside a ```python``` block. No markdown outside it.\n"
        "The code MUST run without errors when executed as `python3 solution.py`."
    )


def build_problematizer_prompt(problem: str, current_code: str) -> str:
    return PROBLEMATIZER_PROMPT.format(
        problem=problem, current_code=current_code
    )


def build_meta_agent_prompt(self_model: str) -> str:
    return META_AGENT_PROMPT.format(self_model=self_model)


SKILL_COMPRESSION_PROMPT = """\
Given the following memory entries from recent optimization rounds:

{solution_entries}
{refinement_entries}
{execution_entries}

Extract (total under 500 tokens):
(a) 2 reusable code patterns (procedural, ~200t each)
(b) 3 rules of thumb / guardrails (declarative, ~30t each)
(c) 1 key lesson from the latest failure (episodic, ~100t)

Be concrete. Use code snippets or condition names. Keep total under 500 tokens.
Output as plain text. Do not use JSON."""


def build_skill_compression_prompt(
    dmf_solutions: dict,
    dmf_refinements: dict,
    dmf_errors: dict,
) -> str:
    sol_entries = "\n".join(
        f"- Round {v.get('round', '?')}: score={v.get('score', 0.0)} tag={v.get('tag', '?')}"
        for v in sorted(dmf_solutions.values(), key=lambda x: x.get('score', 0), reverse=True)[:5]
    )
    ref_entries = "\n".join(
        f"- delta={v.get('delta', 0.0):.4f} conf={v.get('confidence', 0.0):.2f} count={v.get('count', '?')}"
        for v in sorted(dmf_refinements.values(), key=lambda x: x.get('confidence', 0), reverse=True)[:5]
    )
    err_entries = "\n".join(
        f"- error={v.get('error', '?')[:80]} count={v.get('count', '?')}"
        for v in sorted(dmf_errors.values(), key=lambda x: x.get('count', 0), reverse=True)[:3]
    )
    return SKILL_COMPRESSION_PROMPT.format(
        solution_entries=sol_entries or "(none)",
        refinement_entries=ref_entries or "(none)",
        execution_entries=err_entries or "(none)",
    )


def build_anomaly_debate_prompts(
    score: float, mean: float, problem: str
) -> tuple[str, str]:
    a = ANOMALY_DEBATE_A.format(score=score, mean=mean, problem=problem)
    b = ANOMALY_DEBATE_B.format(score=score, mean=mean, problem=problem)
    return a, b
