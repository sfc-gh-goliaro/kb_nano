"""LLM API helper for the Corvo internal endpoint.

Provides both async and synchronous interfaces for calling Claude models.
"""

from __future__ import annotations

import asyncio
import functools
import os
import shutil
import ssl
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import aiohttp

URL_CORVO_COMPLETE_XP = (
    "https://corvoproxy.preprodc1.us-west-2.aws-dev.app.snowflake.com"
    "/v1/textcompletion_xp"
)

_HEADERS = {
    "sf-external-function-signature": (
        "(MODEL VARCHAR, MESSAGES ARRAY, OPTIONS OBJECT)"
    ),
    "sf-external-function-name": "TRY_COMPLETE$V2",
    "sf-ml-account-hash": "internal-eval",
    "sf-ml-enabled-cross-regions": "ANY_REGION",
}


async def _do_request(
    session: aiohttp.ClientSession,
    payload: dict,
) -> str | None:
    timeout = aiohttp.ClientTimeout(total=300)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    async with session.post(
        url=URL_CORVO_COMPLETE_XP,
        json=payload,
        headers=_HEADERS,
        timeout=timeout,
        ssl=ssl_context,
    ) as response:
        response_text = await response.text()
        if response.status != 200:
            raise RuntimeError(
                f"Corvo request failed: {response.status} - {response_text[:500]}"
            )

        resp_json = await response.json()
        data = resp_json.get("data", [])
        if not data:
            raise RuntimeError(f"Empty response data: {response_text[:500]}")

        result = data[0][1]
        if isinstance(result, dict) and "choices" in result:
            choice = result["choices"][0]
            if isinstance(choice, dict):
                if "messages" in choice:
                    return choice["messages"]
                if "message" in choice and isinstance(choice["message"], dict):
                    return choice["message"].get("content", "")
                if "text" in choice:
                    return choice["text"]

        raise RuntimeError(f"Unexpected response format: {response_text[:500]}")


@functools.cache
def _claude_bin() -> str | None:
    """Absolute path to the Claude Code CLI.

    ``shutil.which`` alone is not enough: the CLI is installed under nvm, whose
    bin directory is added by an interactive shell profile.  A background job,
    cron run, or subprocess with a minimal environment therefore fails with
    "claude: No such file or directory" and the backend silently falls back to
    the unreachable Corvo endpoint.
    """
    found = shutil.which("claude")
    if found:
        return found
    for cand in sorted(Path.home().glob(".nvm/versions/node/*/bin/claude")):
        if os.access(cand, os.X_OK):
            return str(cand)
    return None


def _claude_cli_available() -> bool:
    return _claude_bin() is not None


def call_llm_claude_cli(
    prompt: str,
    model_name: str = "claude-opus-4-6",
    system: str | None = None,
    timeout: int = 1500,
) -> str:
    """Generate via the locally installed Claude Code CLI.

    Backend for environments without access to the Corvo proxy (which is
    reachable only from inside Snowflake).  ``claude -p`` runs headless and
    prints the completion on stdout.

    The prompt goes in on **stdin**, not argv: a kernel-generation prompt embeds
    the whole baseline module and easily reaches several kilobytes, which is
    fragile as a command-line argument.  No tool flags are passed -- the task is
    pure text generation, and an empty ``--allowedTools ""`` was observed to
    make the CLI hang until the timeout.
    """
    # A prompt whose *text* embeds source code to rewrite reliably aborts the
    # CLI's stream in this environment ("terminal_reason": "aborted_streaming",
    # main model 0 tokens).  Handing the same content over as a *file* the model
    # reads with its Read tool completes normally, so the task is staged on disk:
    # instructions in task.md, answer written to out.py.
    workdir = tempfile.mkdtemp(prefix="fk_llm_")
    task_path = os.path.join(workdir, "task.md")
    out_path = os.path.join(workdir, "out.py")
    with open(task_path, "w") as fh:
        fh.write(prompt)

    instruction = (
        "Read the file task.md in this directory and carry out the instructions "
        "in it. Write your complete answer to out.py -- the contents of the "
        "single Python code block it asks for, with no ``` fences and no "
        "commentary. out.py must be valid, self-contained Python. "
        "Reply with just DONE when finished."
    )
    cmd = [
        _claude_bin() or "claude", "-p", instruction,
        "--model", model_name,
        "--allowedTools", "Read,Write",
        "--permission-mode", "acceptEdits",
    ]
    if system:
        cmd += ["--append-system-prompt", system]
    # The CLI is intermittently slow on these tasks (observed 661s success and
    # 900s hangs for the same prompt), so a single timeout is retried once
    # before giving up.
    proc = None
    for attempt in range(2):
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
                cwd=workdir,
            )
            break
        except subprocess.TimeoutExpired:
            if attempt == 1:
                raise
            if os.path.exists(out_path) and open(out_path).read().strip():
                break  # it wrote the answer before stalling on teardown
    if os.path.exists(out_path):
        code = open(out_path).read().strip()
        if code:
            # The agent's extractor expects a fenced block.
            return f"```python\n{code}\n```"
    if proc is not None and proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed ({proc.returncode}): "
            f"{(proc.stderr or proc.stdout)[:500]}"
        )
    out = (proc.stdout if proc is not None else '').strip()
    if not out:
        raise RuntimeError("claude CLI returned empty output")
    return out


def _backend() -> str:
    """Which LLM backend to use: 'claude-cli' or 'corvo'."""
    choice = os.environ.get("FASTKERNELS_LLM_BACKEND", "").strip().lower()
    if choice:
        return choice
    return "claude-cli" if _claude_cli_available() else "corvo"


async def call_llm_async(
    prompt: str,
    model_name: str = "claude-opus-4-6",
    max_tokens: int = 8192,
    temperature: float = 0.0,
    system: str | None = None,
    session: aiohttp.ClientSession | None = None,
) -> str:
    """Send a prompt to the LLM and return the generated text.

    Raises RuntimeError on failure.
    """
    if _backend() == "claude-cli":
        return await asyncio.to_thread(
            call_llm_claude_cli, prompt, model_name, system,
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    options = {"max_tokens": max_tokens, "temperature": temperature}
    payload = {"data": [[0, model_name, messages, options]]}

    if session is None:
        async with aiohttp.ClientSession() as new_session:
            result = await _do_request(new_session, payload)
    else:
        result = await _do_request(session, payload)

    if result is None:
        raise RuntimeError("LLM returned None")
    return result


def call_llm(
    prompt: str,
    model_name: str = "claude-opus-4-6",
    max_tokens: int = 8192,
    temperature: float = 0.0,
    system: str | None = None,
) -> str:
    """Synchronous wrapper around call_llm_async."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(
                call_llm_async(prompt, model_name, max_tokens, temperature, system)
            )
        finally:
            new_loop.close()
    else:
        return asyncio.run(
            call_llm_async(prompt, model_name, max_tokens, temperature, system)
        )
