"""Utility for extracting executable code from LLM responses.

Handles cases where LLM produces:
- Markdown code blocks (```python ... ```)
- Explanatory text before/after code
- Mixed text and code
"""

import re
from typing import Optional


def extract_python_code(llm_response: str) -> str:
    """Extract executable Python code from LLM response.

    Args:
        llm_response: Raw response from LLM that may contain explanatory text

    Returns:
        Cleaned Python code ready for execution

    Strategy:
    1. Try to extract from markdown code blocks first
    2. If no blocks, look for lines that appear to be Python code
    3. Strip any explanatory text before/after code
    """
    if not llm_response or not llm_response.strip():
        return ""

    # Strategy 1: Extract from markdown code blocks
    if "```python" in llm_response:
        # Find all python code blocks
        matches = re.findall(r'```python\s*(.*?)```', llm_response, re.DOTALL)
        if matches:
            code = matches[0].strip()
            # Validate that the code block actually contains Python, not apologies
            if _is_refusal_or_apology(code):
                return ""  # Return empty string to signal no valid code
            return code

    if "```" in llm_response:
        # Generic code block
        matches = re.findall(r'```\s*(.*?)```', llm_response, re.DOTALL)
        if matches:
            code = matches[0].strip()
            if _is_refusal_or_apology(code):
                return ""
            return code

    # Strategy 2: No markdown blocks - extract lines that look like Python
    lines = llm_response.split('\n')
    code_lines = []
    in_code = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines at the start
        if not code_lines and not stripped:
            continue

        # Check if line looks like Python code
        is_code_line = (
            # Imports
            stripped.startswith('import ') or
            stripped.startswith('from ') or
            # Common Python patterns
            '=' in stripped or
            stripped.startswith('def ') or
            stripped.startswith('class ') or
            stripped.startswith('if ') or
            stripped.startswith('for ') or
            stripped.startswith('while ') or
            stripped.startswith('with ') or
            stripped.startswith('try:') or
            stripped.startswith('except') or
            stripped.startswith('print(') or
            # Continuation from previous code
            (in_code and (stripped.startswith('    ') or stripped.startswith('\t')))
        )

        # Check if line is likely explanatory text
        is_explanation = (
            stripped.startswith('The ') or
            stripped.startswith('This ') or
            stripped.startswith('Here') or
            stripped.startswith('Let') or
            stripped.startswith('We ') or
            stripped.startswith('Now ') or
            stripped.startswith('First') or
            stripped.startswith('Next') or
            stripped.endswith(':') and not stripped.endswith('::') and not any(kw in stripped for kw in ['if', 'for', 'while', 'def', 'class', 'try', 'except', 'with'])
        )

        if is_code_line and not is_explanation:
            code_lines.append(line)
            in_code = True
        elif in_code and stripped == '':
            # Allow blank lines within code
            code_lines.append(line)
        elif in_code and not is_code_line:
            # Hit non-code after code started - stop
            break

    code = '\n'.join(code_lines).strip()

    # Strategy 3: Last resort - if we got nothing, return original but warn
    if not code:
        # Check if entire response might be code (no prose)
        if _looks_like_pure_code(llm_response):
            return llm_response.strip()
        return llm_response.strip()  # Return as-is and let execution fail with clear error

    return code


def _looks_like_pure_code(text: str) -> bool:
    """Check if text appears to be pure code without explanatory prose."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return False

    # Count lines that look like code vs prose
    code_indicators = 0
    prose_indicators = 0

    for line in lines:
        if line.startswith('#'):
            continue  # Skip comments

        # Code indicators
        if any(line.startswith(kw) for kw in ['import', 'from', 'def', 'class', 'if', 'for', 'while', 'with', 'try']):
            code_indicators += 1
        elif '=' in line or line.startswith('print('):
            code_indicators += 1

        # Prose indicators
        if line[0].isupper() and line.endswith('.') and ' ' in line:
            prose_indicators += 1

    return code_indicators > prose_indicators


def _is_refusal_or_apology(text: str) -> bool:
    """Check if text is a refusal or apology rather than actual code.

    Args:
        text: Code or text to check

    Returns:
        True if text appears to be a refusal/apology, False otherwise
    """
    text_lower = text.lower().strip()

    refusal_patterns = [
        "i'm sorry",
        "i cannot",
        "i can't",
        "impossible to",
        "without the actual data",
        "without this information",
        "without the data",
        "please provide",
        "i don't have",
        "i do not have",
        "as a text-based ai",
        "as an ai",
        "pseudo-code",
        "pseudo code",
        "general approach",
        "here is a general",
        "can't be run directly",
        "won't run in python",
    ]

    return any(pattern in text_lower for pattern in refusal_patterns)


def validate_python_syntax(code: str) -> tuple[bool, Optional[str]]:
    """Check if extracted code is valid Python syntax.

    Args:
        code: Python code string

    Returns:
        (is_valid, error_message)
    """
    try:
        compile(code, '<string>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, str(e)
