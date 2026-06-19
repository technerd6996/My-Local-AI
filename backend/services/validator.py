import re
import ast
import json
import subprocess
import os
import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """
    Finds every fenced code block in the answer and returns
    (language_tag, code) pairs. Language tag comes from whatever
    follows the opening ``` — e.g. ```python, ```hcl, ```powershell.
    If the model didn't tag it, lang comes back as "unknown".
    """
    blocks = []
    for match in CODE_BLOCK_PATTERN.finditer(text):
        lang = match.group(1).strip().lower() or "unknown"
        code = match.group(2).strip()
        if code:
            blocks.append((lang, code))
    return blocks


def validate_python(code: str) -> dict:
    """ast.parse never executes the code — it only builds a syntax tree.
    A SyntaxError here means the code genuinely won't run."""
    try:
        ast.parse(code)
        return {"valid": True, "method": "ast.parse", "error": None}
    except SyntaxError as e:
        return {"valid": False, "method": "ast.parse", "error": str(e)}


def validate_json(code: str) -> dict:
    try:
        json.loads(code)
        return {"valid": True, "method": "json.loads", "error": None}
    except json.JSONDecodeError as e:
        return {"valid": False, "method": "json.loads", "error": str(e)}


def validate_yaml(code: str) -> dict:
    try:
        import yaml
        yaml.safe_load(code)
        return {"valid": True, "method": "yaml.safe_load", "error": None}
    except Exception as e:
        return {"valid": False, "method": "yaml.safe_load", "error": str(e)}


def validate_shell(code: str) -> dict:
    """
    -n is syntax-check-only on both bash and sh — nothing in the
    script is ever executed. Tries bash first since most generated
    scripts assume bash features; falls back to sh if bash isn't
    installed in this container.
    """
    for shell in ("bash", "sh"):
        try:
            result = subprocess.run(
                [shell, "-n"], input=code, capture_output=True,
                text=True, timeout=5
            )
            return {
                "valid": result.returncode == 0,
                "method": f"{shell} -n",
                "error": result.stderr.strip() if result.returncode != 0 else None
            }
        except FileNotFoundError:
            continue
    return {"valid": False, "method": "shell", "error": "no shell interpreter found for syntax check"}


# This dict is intentionally small. It's not "the languages we support" —
# it's "the languages we happen to have a real checker installed for."
# Anything not listed here (Terraform, PowerShell, Go, Rust, whatever)
# falls through to the LLM judge instead of being rejected or ignored.
# Adding a new native check later is just adding one entry here.
NATIVE_VALIDATORS = {
    "python": validate_python,
    "py": validate_python,
    "json": validate_json,
    "yaml": validate_yaml,
    "yml": validate_yaml,
    "bash": validate_shell,
    "sh": validate_shell,
    "shell": validate_shell,
}


async def llm_judge(query: str, answer: str, judge_model: str, is_code: bool = False) -> dict:
    """
    Has a local model critique another model's answer.
    judge_model is always the OTHER local model from whichever drafted
    the answer (qwen3:8b drafts -> deepseek-r1:7b judges, and vice versa).
    Cross-model judging matters because a model grading its own output
    tends to rubber-stamp it — a different model is a meaningfully
    different (if imperfect) check.
    """
    if is_code:
        rubric = (
            "You are reviewing code for correctness. Check for syntax errors, "
            "logic errors, missing error handling, and whether it actually "
            "does what was asked."
        )
    else:
        rubric = (
            "You are reviewing a written answer for accuracy and completeness. "
            "Check whether it actually answers the question asked, whether it "
            "contradicts itself, and whether it contains suspiciously specific "
            "claims (exact version numbers, named APIs, named products) that "
            "may be hallucinated rather than verified facts."
        )

    judge_prompt = f"""{rubric}

Original question:
{query}

Answer to review:
{answer}

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"verdict": "pass" or "fail", "reason": "one sentence explaining why"}}"""

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                "model": judge_model,
                "messages": [{"role": "user", "content": judge_prompt}],
                "stream": False,
                "temperature": 0.1,
            }
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]

    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip()).strip()

    try:
        result = json.loads(cleaned)
        verdict = result.get("verdict", "fail").lower()
        reason = result.get("reason", "no reason given")
        return {
            "valid": verdict == "pass",
            "method": f"llm_judge:{judge_model}",
            "error": None if verdict == "pass" else reason
        }
    except (json.JSONDecodeError, KeyError):
        # If the judge's own response can't be parsed, fail safe rather
        # than assume the answer was fine — we'd rather over-flag than
        # silently pass something we never actually verified.
        return {
            "valid": False,
            "method": f"llm_judge:{judge_model}",
            "error": f"could not parse judge response: {cleaned[:100]}"
        }


async def validate_answer(query: str, answer: str, drafting_model: str) -> dict:
    """
    Entry point. Decides HOW to validate based on what's actually in
    the answer — not on a pre-decided task category.
    """
    code_blocks = extract_code_blocks(answer)
    judge_model = "deepseek-r1:7b" if drafting_model == "qwen3:8b" else "qwen3:8b"

    if code_blocks:
        # Validate the largest block — usually the real answer,
        # smaller ones tend to be illustrative snippets.
        lang, code = max(code_blocks, key=lambda b: len(b[1]))

        if lang in NATIVE_VALIDATORS:
            result = NATIVE_VALIDATORS[lang](code)
        else:
            result = await llm_judge(query, answer, judge_model, is_code=True)
        result["code_lang"] = lang
        return result

    result = await llm_judge(query, answer, judge_model, is_code=False)
    result["code_lang"] = None
    return result


def escalate_stub(query: str, local_answer: str, validation: dict) -> str:
    """
    Phase 3 placeholder. Phase 4 replaces the BODY of this function with
    an actual Claude/ChatGPT call. For now, failing validation just gets
    flagged clearly instead of silently shown as if it were trustworthy.
    """
    warning = (
        f"⚠️ This answer did not pass validation "
        f"({validation['method']}: {validation.get('error', 'unspecified')}). "
        f"Cloud escalation isn't wired up yet — that's Phase 4. "
        f"Treat this answer with extra caution.\n\n---\n\n"
    )
    return warning + local_answer