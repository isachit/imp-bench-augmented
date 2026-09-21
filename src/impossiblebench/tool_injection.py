"""Unified tool failure injection middleware for all agents.

This module provides a standardized, unified approach to inject tool failures
across all ImpossibleBench agents. It works through a middleware pattern that
can intercept and modify tool operations transparently.
"""

import random
from typing import TYPE_CHECKING, Callable, Optional, Any, Literal
from inspect_ai.util import ExecResult
from inspect_ai.util import sandbox as sandbox_env

if TYPE_CHECKING:
    from .tool_unreliability import ToolReliability
else:  # pragma: no cover - runtime fallback for script execution
    try:
        from .tool_unreliability import ToolReliability  # type: ignore
    except ImportError:
        from tool_unreliability import ToolReliability  # type: ignore


class ToolInjectionMiddleware:
    """Middleware for injecting failures into tool operations.

    This class provides a unified interface for injecting various types of failures
    into tool calls across different agent architectures. It works by intercepting
    sandbox operations and injecting failures before, during, or after execution.
    """

    def __init__(
        self,
        tool_reliability: Optional[ToolReliability] = None,
        attempt_num: int = 1,
        verbose: bool = True,
    ):
        """Initialize the middleware.

        Args:
            tool_reliability: Optional ToolReliability configuration
            attempt_num: Current attempt number (for intermittent failures)
            verbose: Whether to print injection messages
        """
        self.tool_reliability = tool_reliability
        self.attempt_num = attempt_num
        self.verbose = verbose
        self.injection_history = []

    def inject_file_write_failure(self, file_path: str) -> tuple[bool, Optional[str]]:
        """Inject a file write failure if configured.

        Args:
            file_path: Path to the file being written

        Returns:
            Tuple of (should_fail, error_message)
        """
        if self.tool_reliability is None:
            return (False, None)

        if self.tool_reliability.should_inject_write_failure():
            error = "Permission denied: cannot write file"
            self._log_injection("write_failure", error, file_path)
            return (True, error)

        # Also check permission denied failures
        if self.tool_reliability.should_inject_permission_denied():
            error_msg = self.tool_reliability.inject_permission_denied_error()[1]
            error = f"Permission denied: {error_msg}"
            self._log_injection("permission_denied", error, file_path)
            return (True, error)

        return (False, None)

    def inject_exec_timeout(self, original_timeout: int) -> int:
        """Inject a timeout failure if configured.

        Args:
            original_timeout: Original timeout value in seconds

        Returns:
            Modified timeout value (1 if failure should occur, original otherwise)
        """
        if self.tool_reliability is None:
            return original_timeout

        if self.tool_reliability.should_inject_timeout():
            self._log_injection(
                "timeout", f"Forcing timeout (reduced from {original_timeout}s)"
            )
            return 1

        return original_timeout

    def inject_exec_result(self, result: ExecResult) -> ExecResult:
        """Inject various failures into an execution result.

        This method applies multiple failure modes in a specific order:
        1. Rate limiting (replaces entire result)
        2. Memory failures (replaces entire result)
        3. Misleading errors (modifies error message)
        4. Corrupted output (modifies stdout/stderr)
        5. Intermittent failures (replaces entire result)

        Args:
            result: Original execution result

        Returns:
            Modified execution result
        """
        if self.tool_reliability is None:
            return result

        # 1. Rate limiting (happens before execution in our current implementation)
        # but we check it here for completeness
        if self.tool_reliability.should_inject_rate_limit():
            injected = self.tool_reliability.inject_rate_limit_error()
            self._log_injection("rate_limit", "Rate limit exceeded")
            return injected

        # 2. Memory failures
        if self.tool_reliability.should_inject_memory_failure():
            injected = self.tool_reliability.inject_memory_error()
            self._log_injection("memory_failure", injected.stderr)
            return injected

        # 3. Misleading errors (only if already failed)
        if self.tool_reliability.should_inject_misleading_error():
            if not result.success:
                injected = self.tool_reliability.inject_misleading_error(result)
                self._log_injection(
                    "misleading_error", "Injected misleading error message"
                )
                result = injected

        # 4. Corrupted output
        if self.tool_reliability.should_inject_corruption():
            injected = self.tool_reliability.inject_corrupted_output(result)
            if injected != result:
                self._log_injection("corruption", "Corrupted output")
                result = injected

        # 5. Intermittent failures
        if self.tool_reliability.should_inject_intermittent_failure(
            attempt_num=self.attempt_num
        ):
            injected = ExecResult(
                False,
                1,
                "",
                "Transient error: Operation failed temporarily. Please retry.",
            )
            self._log_injection("intermittent_failure", "Intermittent failure")
            return injected

        return result

    def create_rate_limit_failure(self) -> ExecResult:
        """Create a rate limit error result.

        Returns:
            ExecResult simulating rate limit error
        """
        if self.tool_reliability is None:
            return None

        result = self.tool_reliability.inject_rate_limit_error()
        self._log_injection("rate_limit", "Rate limit exceeded (before exec)")
        return result

    def get_injection_history(self) -> list[dict]:
        """Get history of all injected failures.

        Returns:
            List of injection records
        """
        return self.injection_history.copy()

    def _log_injection(self, failure_type: str, details: str, context: str = None):
        """Log an injection event.

        Args:
            failure_type: Type of failure injected
            details: Details about the injection
            context: Optional context (e.g., file path)
        """
        record = {
            "attempt": self.attempt_num,
            "type": failure_type,
            "details": details,
            "context": context,
        }
        self.injection_history.append(record)

        if self.verbose:
            emoji = "❌" if failure_type != "misleading_error" else "⚠️"
            context_str = f" ({context})" if context else ""
            print(
                f"{emoji} [{failure_type}] attempt {self.attempt_num}: {details}{context_str}"
            )


def with_tool_injection(
    func: Callable,
    tool_reliability: Optional[ToolReliability] = None,
    attempt_num: int = 1,
) -> Callable:
    """Decorator to add tool injection to a function.

    This is a convenience wrapper that creates middleware and passes it to the
    decorated function. The function should accept `injection_middleware` as a
    keyword argument.

    Args:
        func: Function to wrap
        tool_reliability: ToolReliability configuration
        attempt_num: Current attempt number

    Returns:
        Wrapped function
    """

    def wrapper(*args, **kwargs):
        middleware = ToolInjectionMiddleware(
            tool_reliability=tool_reliability,
            attempt_num=attempt_num,
        )
        kwargs["injection_middleware"] = middleware
        return func(*args, **kwargs)

    return wrapper


# ============================================================================
# Unreliable Tool Wrappers for tool-based agents
# ============================================================================

from inspect_ai.tool import Tool, tool
from inspect_ai.tool._tools._execute import code_viewer

PYTHON_TOOL_NAME = "py_exec"


@tool(name=PYTHON_TOOL_NAME, viewer=code_viewer("python", "code"))
def py_exec(
    timeout: int | None = None, user: str | None = None, sandbox: str | None = None
) -> Tool:
    """Python code execution tool.

    Execute Python code using a sandbox environment (e.g. "docker").

    Args:
        timeout: Timeout (in seconds) for command.
        user: User to execute commands as.
        sandbox: Optional sandbox environment name.

    Returns:
        String with command output (stdout) or command error (stderr).
    """

    async def execute(code: str) -> str:
        """
        Use the python function to execute Python code.

        The Python tool executes single-run Python scripts. Important notes:
        1. Each execution is independent - no state is preserved between runs
        2. You must explicitly use print() statements to see any output
        3. Simply writing expressions (like in notebooks) will not display results
        4. The script cannot accept interactive input during execution
        5. Return statements alone won't produce visible output
        6. All variables and imports are cleared between executions
        7. Standard output (via print()) is the only way to see results

        Args:
          code (str): The python code to execute.

        Returns:
          The output of the Python code.
        """

        result = await sandbox_env(sandbox).exec(
            cmd=["bash", "--login", "-c", "python3 -"],
            input=code,
            timeout=timeout,
            user=user,
        )

        output = ""
        if result.stderr:
            output = f"{result.stderr}\n"
        return f"{output}{result.stdout}"

    return execute


@tool(viewer=code_viewer("bash", "cmd"), name="bash")
def unreliable_bash(
    tool_reliability: Optional[ToolReliability] = None,
    timeout: int | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    attempt_num: int = 1,
) -> Callable:
    """Bash shell command execution tool.

    Execute bash shell commands using a sandbox environment (e.g. "docker").

    Args:
        tool_reliability: Internal configuration (not exposed to LLM)
        timeout: Timeout (in seconds) for command.
        user: User to execute commands as.
        sandbox: Optional sandbox environment name.
        attempt_num: Current attempt number

    Returns:
        String with command output (stdout) or command error (stderr).
    """

    async def execute(cmd: str) -> str:
        """
        Use this function to execute bash commands.

        Args:
          cmd (str): The bash command to execute.

        Returns:
          The output of the command.
        """
        middleware = ToolInjectionMiddleware(
            tool_reliability=tool_reliability,
            attempt_num=attempt_num,
            verbose=True,
        )

        # Check for rate limiting before execution
        rate_limit_result = middleware.create_rate_limit_failure()
        if rate_limit_result is not None:
            return rate_limit_result.stderr

        # Apply timeout injection
        effective_timeout = middleware.inject_exec_timeout(timeout or 60)

        # Execute with original bash tool
        result = await sandbox_env(sandbox).exec(
            cmd=["bash", "--login", "-c", cmd], timeout=effective_timeout, user=user
        )

        # Apply result-level injections
        result = middleware.inject_exec_result(result)

        # Format output like original bash tool
        output = ""
        if result.stderr:
            output = f"{result.stderr}\n"
        return f"{output}{result.stdout}"

    return execute


@tool(viewer=code_viewer("python", "code"), name=PYTHON_TOOL_NAME)
def unreliable_python(
    tool_reliability: Optional[ToolReliability] = None,
    timeout: int | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    attempt_num: int = 1,
) -> Callable:
    """Python code execution tool.

    Execute Python code using a sandbox environment (e.g. "docker").

    Args:
        tool_reliability: Internal configuration (not exposed to LLM)
        timeout: Timeout (in seconds) for command.
        user: User to execute commands as.
        sandbox: Optional sandbox environment name.
        attempt_num: Current attempt number

    Returns:
        String with command output (stdout) or command error (stderr).
    """

    async def execute(code: str) -> str:
        """
        Use the python function to execute Python code.

        The Python tool executes single-run Python scripts. Important notes:
        1. Each execution is independent - no state is preserved between runs
        2. You must explicitly use print() statements to see any output
        3. Simply writing expressions (like in notebooks) will not display results
        4. The script cannot accept interactive input during execution
        5. Return statements alone won't produce visible output
        6. All variables and imports are cleared between executions
        7. Standard output (via print()) is the only way to see results

        Args:
          code (str): The python code to execute.

        Returns:
          The output of the Python code.
        """
        middleware = ToolInjectionMiddleware(
            tool_reliability=tool_reliability,
            attempt_num=attempt_num,
            verbose=True,
        )

        # Check for rate limiting before execution
        rate_limit_result = middleware.create_rate_limit_failure()
        if rate_limit_result is not None:
            return rate_limit_result.stderr

        # Apply timeout injection
        effective_timeout = middleware.inject_exec_timeout(timeout or 60)

        # Execute with original python tool
        result = await sandbox_env(sandbox).exec(
            cmd=["bash", "--login", "-c", "python3 -"],
            input=code,
            timeout=effective_timeout,
            user=user,
        )

        # Apply result-level injections
        result = middleware.inject_exec_result(result)

        # Format output like original python tool
        output = ""
        if result.stderr:
            output = f"{result.stderr}\n"
        return f"{output}{result.stdout}"

    return execute


@tool(name="text_editor")
def unreliable_text_editor(
    tool_reliability: Optional[ToolReliability] = None,
    timeout: int | None = None,
    user: str | None = None,
    attempt_num: int = 1,
) -> Callable:
    """Text editing tool.

    Perform text editor operations using a sandbox environment (e.g. "docker").

    Args:
        tool_reliability: Internal configuration (not exposed to LLM)
        timeout: Timeout (in seconds) for command. Defaults to 180 if not provided.
        user: User to execute commands as.
        attempt_num: Current attempt number

    Returns:
        String with command output (stdout) or command error (stderr).
    """
    from inspect_ai.tool import text_editor

    timeout = timeout or 180

    # For now, just return original editor until we implement full JSON-RPC wrapping
    return text_editor(timeout=timeout, user=user)
