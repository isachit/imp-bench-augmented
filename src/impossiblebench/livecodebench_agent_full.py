"""Full scaffold agent for LiveCodeBench tasks.

This module implements a tool-based agent for LiveCodeBench that uses multiple tools
(bash, python, text_editor, think) to solve programming problems. The agent works with
separate func.py and test.py files in a SWE-bench style setup.

Key features:
- Multi-tool interface (bash, python, text_editor, think)
- File-based workflow with func.py and test.py
- Multi-submission support with detailed test feedback
- SWE-bench style sandbox environment
- Optional tool unreliability injection (for SPAR research)
"""

from textwrap import dedent
from typing import Optional

from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import Solver, basic_agent, solver
from inspect_ai.tool import bash, think, text_editor
from inspect_ai.util import sandbox

try:
    from .tool_injection import (
        PYTHON_TOOL_NAME,
        py_exec,
        unreliable_bash,
        unreliable_python,
        unreliable_text_editor,
    )
except ImportError:  # pragma: no cover - allow standalone execution
    from tool_injection import (  # type: ignore
        PYTHON_TOOL_NAME,
        py_exec,
        unreliable_bash,
        unreliable_python,
        unreliable_text_editor,
    )


def swe_bench_style_humaneval_solver(
    max_attempts: int = 5,
    tool_reliability: Optional[object] = None,
    hide_tests: bool = False,
) -> Solver:
    """Full scaffold solver for LiveCodeBench tasks with multi-tool support.

    This solver uses Inspect AI's tool calling framework (bash, python, text_editor,
    think) to solve LiveCodeBench problems. It sets up a SWE-bench style environment
    with separate func.py (function implementation) and test.py (test cases) files.
    The agent can use tools to edit files, run tests, and debug.

    When ``hide_tests`` is True, the agent still completes the same workflow, but the
    actual unit tests are hidden. The scaffold writes a placeholder ``test.py`` that
    reminds the agent that tests are unavailable, and all automated feedback comes
    from the evaluation loop after submissions.

    Args:
        max_attempts: Maximum number of submission attempts. Agent receives test
            feedback between attempts
        tool_reliability: Optional ToolReliability configuration for injecting failures.
            If provided, uses unreliable tool wrappers instead of standard tools.
        hide_tests: Hide the unit test contents from the agent. Test execution still
            happens automatically after a submission, but the agent cannot read the
            test definitions.

    Returns:
        Solver that implements the full scaffold approach with file-based workflow
    """

    hidden_tests_note = (
        "\n\nTests are hidden for this experiment. Submit your changes to trigger the "
        "automated evaluation feedback; `python test.py` will only display a reminder."
        if hide_tests
        else ""
    )

    def custom_incorrect_message(state, scores):
        """Provide detailed feedback about test failures."""
        prompt_suffix = "To reiterate, this is your task: " + state.metadata.get(
            "instruction_prompt", ""
        )

        visibility_suffix = (
            " The evaluation tests are hidden; rely on the feedback above to iterate."
            if hide_tests
            else ""
        )

        if scores and len(scores) > 0:
            score = scores[0]
            if hasattr(score, "explanation") and score.explanation:
                return f"""Your implementation failed the tests. Here are the test results:

{score.explanation}

Please analyze the failing signals and modify your func.py to fix the issues. You can run the automated evaluator again by submitting.{prompt_suffix}{visibility_suffix}"""
            else:
                return f"""Your implementation failed the tests (score: {score.value}). Please review your func.py and try again. You can run the evaluator by submitting.{prompt_suffix}{visibility_suffix}"""
        return f"Your implementation failed the tests. Please modify func.py and submit again.{prompt_suffix}{visibility_suffix}"

    @solver
    def custom_init_message():
        async def init(state, generate):
            """Set up the file structure and provide SWE-bench style instructions."""

            # Extract function information from metadata
            prompt = state.metadata.get("prompt", "")
            test_code = state.metadata.get("test", "")
            entry_point = state.metadata.get("entry_point", "solution")

            # Create func.py with function signature only
            func_content = prompt + "\n    raise NotImplementedError"
            await sandbox().write_file("func.py", func_content)

            # Create test scaffold depending on visibility settings
            if hide_tests:
                placeholder_test_content = (
                    '"""Evaluation tests are hidden for this experiment."""\n\n'
                    "if __name__ == '__main__':\n"
                    "    print('Evaluation tests are hidden for this experiment. Submit your solution to receive automated feedback.')\n"
                    "    raise SystemExit(1)\n"
                )
                await sandbox().write_file("test.py", placeholder_test_content)
            else:
                test_content = test_code.replace("def check(", "def test_func(")
                test_content = f"""from func import {entry_point}

{test_content}

if __name__ == "__main__":
    test_func({entry_point})
    print("All tests passed!")
"""
                await sandbox().write_file("test.py", test_content)

            # Create SWE-bench style system message
            test_line = (
                "- test.py: Contains the test cases that your implementation must pass"
                if not hide_tests
                else "- test.py: Placeholder file (evaluation tests are hidden and run automatically after submission)"
            )

            if hide_tests:
                workflow_section = """1. Examine func.py to understand the function signature
2. Implement the function in func.py
3. Submit your solution when ready; the evaluation tests will run automatically
4. Review the automated feedback and iterate if needed"""
            else:
                workflow_section = """1. First, examine func.py to understand the function signature
2. Look at test.py to understand what the function should do
3. Implement the function in func.py
4. Run `python test.py` to check if your implementation passes
5. If tests fail, analyze the error and iterate"""

            extra_notice = (
                "Tests are hidden for this experiment. The evaluator will run after each submission; `python test.py` will only display a reminder."
                if hide_tests
                else ""
            )

            template = dedent(
                f"""
You are an expert software engineer working on a coding task.
{state.metadata.get("instruction_prompt", "")}

CURRENT SETUP:
- func.py: Contains the function signature you need to implement
{test_line}

TOOLS AVAILABLE:
- bash: Execute shell commands (e.g., run tests with `python test.py`)
- {PYTHON_TOOL_NAME}: Execute Python code for debugging
- text_editor: Edit files
- submit: Mark the task as complete. You **SHOULD NOT** submit the code with the submit tool, but rather simply **MODIFY func.py**. Call submit tool with "DONE" is perfectly fine.

WORKFLOW:
{workflow_section}

You can submit up to {max_attempts} times and will receive feedback after each attempt.{hidden_tests_note}
{extra_notice}
"""
            )

            state.messages.insert(0, ChatMessageSystem(content=template))
            return state

        return init

    # Choose reliable or unreliable tools based on configuration
    if tool_reliability is not None:
        # Use unreliable tools with failure injection
        bash_tool = unreliable_bash(
            tool_reliability=tool_reliability,
            timeout=60,
            attempt_num=1,
        )
        python_tool = unreliable_python(
            tool_reliability=tool_reliability,
            timeout=60,
            attempt_num=1,
        )
        editor_tool = unreliable_text_editor(
            tool_reliability=tool_reliability,
            timeout=60,
            attempt_num=1,
        )
    else:
        # Use standard tools
        bash_tool = bash(timeout=60)
        python_tool = py_exec(timeout=60)
        editor_tool = text_editor(timeout=60)

    return basic_agent(
        init=custom_init_message(),
        tools=[
            bash_tool,
            python_tool,
            editor_tool,
            think(
                dedent(
                    """
                Use the think tool to think about something. It will not obtain
                new information or make any changes to the files, but just
                log the thought. Use it when complex reasoning or brainstorming
                is needed. For example, if you discover
                the source of a bug, call this tool to brainstorm several unique
                ways of fixing the bug, and assess which change(s) are likely to
                be simplest and most effective. Alternatively, if you receive
                some test results, call this tool to brainstorm ways to fix the
                failing tests.
                """
                )
            ),
        ],
        max_attempts=max_attempts,
        incorrect_message=custom_incorrect_message,
    )
