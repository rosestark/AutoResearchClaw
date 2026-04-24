"""ACP (Agent Client Protocol) LLM client via codex exec.

Uses ``codex exec`` for reliable single-process prompt→response cycles.
Each call is stateless (no persistent session), but the pipeline already
provides full context through prior-artifact inclusion in each stage prompt.

Historical note: the original acpx persistent-session approach was abandoned
because the queue-owner process dies between subprocess calls, making
``sessions ensure`` + ``prompt -s`` unreliable (sessions created by one
subprocess invocation are unreachable from the next).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

from researchclaw.llm.client import LLMResponse

logger = logging.getLogger(__name__)

# codex exec output markers (same as acpx output markers)
_DONE_RE = re.compile(r"^\[done\]")
_CLIENT_RE = re.compile(r"^\[client\]")
_ACPX_RE = re.compile(r"^\[acpx\]")
_TOOL_RE = re.compile(r"^\[tool\]")
# codex exec specific markers
_SESSION_RE = re.compile(r"^session id:")
_TOKENS_RE = re.compile(r"^tokens used$")
_MCP_RE = re.compile(r"^mcp startup:")
_ROLE_RE = re.compile(r"^(user|codex|assistant|system)$")


@dataclass
class ACPConfig:
    """Configuration for ACP agent connection."""

    agent: str = "codex"
    cwd: str = "."
    acpx_command: str = ""  # kept for config compat, not used
    session_name: str = "researchclaw"
    timeout_sec: int = 1800  # per-prompt timeout
    # Model override for codex exec. Pin explicitly to avoid model drift
    # across OpenClaw/ChatGPT-account/Codex-CLI default changes (e.g., a
    # default of gpt-5.2-codex is rejected by a ChatGPT-account login).
    # Empty string means "use the default from ~/.codex/config.toml".
    model: str = "gpt-5.4"
    # Reasoning effort override for codex exec. Code-generation / literature
    # screen prompts don't benefit from "high" reasoning; "medium" cuts
    # wall-clock by ~2x while still producing valid multi-file code
    # packages.  Set to empty string to defer to ~/.codex/config.toml.
    reasoning_effort: str = "medium"


def _find_agent_binary(agent: str) -> str | None:
    """Find the agent CLI binary on PATH."""
    return shutil.which(agent)


class ACPClient:
    """LLM client that uses codex exec for reliable prompt→response cycles.

    Each ``.chat()`` call spawns a fresh ``codex exec`` process.  This avoids
    the acpx queue-owner lifecycle bug where persistent sessions become
    unreachable between subprocess invocations.
    """

    def __init__(self, acp_config: ACPConfig) -> None:
        self.config = acp_config
        self._agent_binary: str | None = None

    @classmethod
    def from_rc_config(cls, rc_config: Any) -> ACPClient:
        """Build from a ResearchClaw ``RCConfig``."""
        acp = rc_config.llm.acp
        return cls(ACPConfig(
            agent=acp.agent,
            cwd=acp.cwd,
            acpx_command=getattr(acp, "acpx_command", ""),
            session_name=getattr(acp, "session_name", "researchclaw"),
            timeout_sec=getattr(acp, "timeout_sec", 1800),
            model=getattr(acp, "model", "gpt-5.4") or "gpt-5.4",
            reasoning_effort=getattr(
                acp, "reasoning_effort", "medium"
            ) or "medium",
        ))

    # ------------------------------------------------------------------
    # Public interface (matches LLMClient)
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        system: str | None = None,
        strip_thinking: bool = False,
    ) -> LLMResponse:
        """Send a prompt and return the agent's response."""
        prompt_text = self._messages_to_prompt(messages, system=system)
        content = self._send_prompt(prompt_text)
        if strip_thinking:
            from researchclaw.utils.thinking_tags import strip_thinking_tags
            content = strip_thinking_tags(content)
        return LLMResponse(
            content=content,
            model=f"acp:{self.config.agent}",
            finish_reason="stop",
        )

    def preflight(self) -> tuple[bool, str]:
        """Check that the agent CLI is available."""
        agent = self.config.agent
        binary = _find_agent_binary(agent)
        if not binary:
            return False, f"ACP agent CLI not found: {agent!r} (not on PATH)"
        self._agent_binary = binary

        # Quick liveness check
        try:
            result = subprocess.run(
                [binary, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            version = result.stdout.strip().splitlines()[0] if result.stdout.strip() else "unknown"
            return True, f"OK - {agent} {version} ready (direct exec mode)"
        except Exception as exc:  # noqa: BLE001
            return False, f"Agent liveness check failed: {exc}"

    def close(self) -> None:
        """No-op — codex exec is stateless."""
        pass

    def __del__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_agent(self) -> str:
        """Resolve the agent binary path (cached)."""
        if self._agent_binary:
            return self._agent_binary
        binary = _find_agent_binary(self.config.agent)
        if not binary:
            raise RuntimeError(f"Agent CLI not found: {self.config.agent!r}")
        self._agent_binary = binary
        return binary

    def _abs_cwd(self) -> str:
        return os.path.abspath(self.config.cwd)

    _MAX_CLI_PROMPT_BYTES = 100_000
    _MAX_RETRIES = 2

    def _send_prompt(self, prompt: str) -> str:
        """Send a prompt via codex exec and return the response text.

        For large prompts, writes to a temp file and passes via stdin.
        Retries on transient failures.
        """
        last_exc: Exception | None = None
        for attempt in range(1 + self._MAX_RETRIES):
            try:
                prompt_bytes = len(prompt.encode("utf-8"))
                if prompt_bytes > self._MAX_CLI_PROMPT_BYTES:
                    return self._send_prompt_via_stdin(prompt)
                return self._send_prompt_cli(prompt)
            except RuntimeError as exc:
                last_exc = exc
                if attempt < self._MAX_RETRIES:
                    logger.warning(
                        "codex exec failed (%s), retrying (attempt %d/%d)...",
                        exc, attempt + 1, self._MAX_RETRIES,
                    )
        raise last_exc  # type: ignore[misc]

    def _send_prompt_cli(self, prompt: str) -> str:
        """Send prompt via codex exec as a CLI argument."""
        binary = self._resolve_agent()
        cwd = self._abs_cwd()

        cmd = [binary, "exec", "--skip-git-repo-check",
               "--sandbox", "read-only"]
        # Pin model explicitly to avoid drift onto an incompatible default
        # (e.g., gpt-5.2-codex is rejected for ChatGPT-account logins).
        if self.config.model:
            cmd.extend(["--model", self.config.model])
        # Pin reasoning effort to keep wall-clock bounded.  High reasoning
        # on ~40KB code-generation prompts regularly exceeds 300s and often
        # exceeds 600s; medium is sufficient and ~2x faster.
        if self.config.reasoning_effort:
            cmd.extend(["-c",
                f'model_reasoning_effort="{self.config.reasoning_effort}"'])
        cmd.append(prompt)

        logger.info("codex exec: sending prompt (%d chars) from %s", len(prompt), cwd)

        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=self.config.timeout_sec,
            cwd=cwd,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Check for git repo requirement
            if "trusted directory" in stderr or "git-repo-check" in stderr:
                # Retry without --skip-git-repo-check for older codex versions
                cmd_fallback = [binary, "exec", prompt]
                result = subprocess.run(
                    cmd_fallback,
                    capture_output=True, text=True,
                    timeout=self.config.timeout_sec,
                    cwd=cwd,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"codex exec failed (exit {result.returncode}): {result.stderr.strip()}"
                    )
            else:
                raise RuntimeError(
                    f"codex exec failed (exit {result.returncode}): {stderr}"
                )

        response = self._extract_response(result.stdout)
        if not response.strip():
            logger.warning(
                "codex exec returned empty response. Stdout len: %d, stderr: %s",
                len(result.stdout),
                result.stderr[:300] if result.stderr else "<empty>",
            )
            raise RuntimeError("codex exec returned empty response")

        return response

    def _send_prompt_via_stdin(self, prompt: str) -> str:
        """Send a large prompt via stdin (pipe to codex exec -)."""
        binary = self._resolve_agent()
        cwd = self._abs_cwd()

        cmd = [binary, "exec", "--skip-git-repo-check",
               "--sandbox", "read-only"]
        if self.config.model:
            cmd.extend(["--model", self.config.model])
        if self.config.reasoning_effort:
            cmd.extend(["-c",
                f'model_reasoning_effort="{self.config.reasoning_effort}"'])
        cmd.append("-")

        logger.info(
            "codex exec: sending large prompt (%d bytes) via stdin from %s",
            len(prompt.encode("utf-8")), cwd,
        )

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True, text=True,
            timeout=self.config.timeout_sec,
            cwd=cwd,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"codex exec (stdin) failed (exit {result.returncode}): {result.stderr.strip()}"
            )

        response = self._extract_response(result.stdout)
        if not response.strip():
            logger.warning(
                "codex exec (stdin) returned empty response. Stdout len: %d",
                len(result.stdout),
            )
            raise RuntimeError("codex exec returned empty response")

        return response

    @staticmethod
    def _extract_response(raw_output: str) -> str:
        """Extract the agent's actual response from codex exec output.

        Strips codex metadata lines (session id, user prompt echo, tool
        blocks, mcp startup, token counts, role labels) and acpx control
        lines if present.
        """
        lines: list[str] = []
        in_tool_block = False
        skip_next_numbers = False

        for line in raw_output.splitlines():
            # Skip codex exec metadata lines
            if _SESSION_RE.match(line):
                continue
            if line.strip() == "--------":
                continue
            if _TOKENS_RE.match(line):
                skip_next_numbers = True
                continue
            if skip_next_numbers:
                # Token count is on the line after "tokens used"
                if line.strip().replace(",", "").isdigit():
                    continue
                skip_next_numbers = False
            if _MCP_RE.match(line):
                continue
            if _ROLE_RE.match(line.strip()):
                continue

            # Skip acpx control lines (if present from legacy path)
            if _DONE_RE.match(line) or _CLIENT_RE.match(line) or _ACPX_RE.match(line):
                in_tool_block = False
                continue
            if _TOOL_RE.match(line):
                in_tool_block = True
                continue

            # Tool blocks have indented continuation lines
            if in_tool_block:
                if line.startswith("  ") or not line.strip():
                    continue
                in_tool_block = False

            # Skip empty lines at start
            if not lines and not line.strip():
                continue
            lines.append(line)

        # Trim trailing empty lines
        while lines and not lines[-1].strip():
            lines.pop()

        # The last line is often the response echoed again — deduplicate
        # (codex exec prints: role label, response text, token count, response text again)
        if len(lines) >= 2 and lines[-1] == lines[-2]:
            lines.pop()

        return "\n".join(lines)

    @staticmethod
    def _messages_to_prompt(
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
    ) -> str:
        """Flatten a chat-messages list into a single text prompt."""
        parts: list[str] = []
        if system:
            parts.append(f"[System]\n{system}")
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "assistant":
                parts.append(f"[Previous Response]\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)
