import asyncio
import shlex
from dataclasses import dataclass


@dataclass
class ToolResult:
    ok: bool
    output: str


class LocalToolbelt:
    """Allowlisted local shell execution for model tool use."""

    def __init__(self, allowed_commands: list[str] | None = None):
        self.allowed_commands = {
            cmd.lower()
            for cmd in (allowed_commands or ["python", "ollama", "git", "mdap-small"])
        }

    async def run(self, shell: str, command: str, timeout_s: int = 20) -> ToolResult:
        exe = self._first_token(command)
        if not exe or exe.lower() not in self.allowed_commands:
            return ToolResult(False, f"blocked command: {exe or '<empty>'}")

        if shell not in ("bash", "powershell"):
            return ToolResult(False, f"unsupported shell: {shell}")

        if shell == "bash":
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-lc",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                "powershell",
                "-NoProfile",
                "-Command",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(False, "tool timeout")

        text = out.decode("utf-8", errors="replace")
        return ToolResult(proc.returncode == 0, text[:4000])

    def _first_token(self, command: str) -> str | None:
        command = command.strip()
        if not command:
            return None
        try:
            return shlex.split(command)[0]
        except Exception:
            return command.split()[0] if command.split() else None
